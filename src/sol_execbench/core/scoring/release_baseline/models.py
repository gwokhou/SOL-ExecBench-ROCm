# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Immutable, deterministic release-baseline evidence contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sol_execbench.core.evidence.checksums import sha256_file as _sha256_file


RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION = "sol_execbench.release_baseline_bundle.v1"
RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION = (
    "sol_execbench.release_baseline_verification.v1"
)
CLASSIFICATIONS = ("official", "derived", "blocked")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a release evidence file."""

    return _sha256_file(path)


def _require_non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_sha256(value: str | None, field: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_positive_finite(value: float, field: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")


@dataclass(frozen=True)
class ReleaseProvenance:
    """Fixed solution and execution identities required for a release rerun."""

    solution: str
    solution_sha256: str
    environment_fingerprint: str | None = None
    timing_policy: str | None = None
    compiler_build_id: str | None = None
    clock_policy: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.solution, "solution")
        _require_sha256(self.solution_sha256, "solution_sha256")
        for field in (
            "environment_fingerprint",
            "timing_policy",
            "compiler_build_id",
            "clock_policy",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_non_empty(value, field)

    def to_dict(self) -> dict[str, str]:
        payload = {
            "solution": self.solution,
            "solution_sha256": self.solution_sha256,
        }
        for field in (
            "environment_fingerprint",
            "timing_policy",
            "compiler_build_id",
            "clock_policy",
        ):
            value = getattr(self, field)
            if value is not None:
                payload[field] = value
        return payload


@dataclass(frozen=True)
class ReleaseBaselineWorkload:
    """One selected-suite workload and its baseline authority classification."""

    definition: str
    workload_uuid: str
    classification: str
    latency_ms: float | None
    blocker_reason_codes: tuple[str, ...]
    trace_ref: str | None = None
    trace_sha256: str | None = None
    bound_ref: str | None = None
    bound_sha256: str | None = None
    hardware_model_ref: str | None = None
    hardware_model_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.definition, "definition")
        _require_non_empty(self.workload_uuid, "workload_uuid")
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(f"classification must be one of {CLASSIFICATIONS}")
        if self.latency_ms is not None:
            _require_positive_finite(self.latency_ms, "latency_ms")
        if not all(
            isinstance(code, str) and code for code in self.blocker_reason_codes
        ):
            raise ValueError("blocker_reason_codes must contain non-empty strings")
        for field in ("trace_sha256", "bound_sha256", "hardware_model_sha256"):
            _require_sha256(getattr(self, field), field)
        for field in ("trace_ref", "bound_ref", "hardware_model_ref"):
            value = getattr(self, field)
            if value is not None:
                _require_non_empty(value, field)

    @property
    def key(self) -> tuple[str, str]:
        return (self.definition, self.workload_uuid)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "classification": self.classification,
            "latency_ms": self.latency_ms,
            "blocker_reason_codes": list(self.blocker_reason_codes),
        }
        for field in (
            "trace_ref",
            "trace_sha256",
            "bound_ref",
            "bound_sha256",
            "hardware_model_ref",
            "hardware_model_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                payload[field] = value
        return payload


@dataclass(frozen=True)
class ReleaseBaselineBundle:
    """Complete selected-suite baseline evidence, with deterministic serialization."""

    release: str
    suite_manifest_ref: str
    suite_manifest_sha256: str
    baseline_artifact_ref: str
    baseline_artifact_sha256: str
    provenance: ReleaseProvenance
    workloads: tuple[ReleaseBaselineWorkload, ...]
    latency_tolerance_rel: float
    schema_version: str = RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in ("release", "suite_manifest_ref", "baseline_artifact_ref"):
            _require_non_empty(getattr(self, field), field)
        _require_sha256(self.suite_manifest_sha256, "suite_manifest_sha256")
        _require_sha256(self.baseline_artifact_sha256, "baseline_artifact_sha256")
        _require_positive_finite(self.latency_tolerance_rel, "latency_tolerance_rel")
        if self.schema_version != RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported release baseline bundle schema_version")
        keys = [workload.key for workload in self.workloads]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate workload key in release baseline bundle")

    @property
    def summary(self) -> dict[str, int]:
        counts = {classification: 0 for classification in CLASSIFICATIONS}
        for workload in self.workloads:
            counts[workload.classification] += 1
        return {"total": len(self.workloads), **counts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release": self.release,
            "suite_manifest_ref": self.suite_manifest_ref,
            "suite_manifest_sha256": self.suite_manifest_sha256,
            "baseline_artifact_ref": self.baseline_artifact_ref,
            "baseline_artifact_sha256": self.baseline_artifact_sha256,
            "provenance": self.provenance.to_dict(),
            "latency_tolerance_rel": self.latency_tolerance_rel,
            "summary": self.summary,
            "workloads": [
                workload.to_dict()
                for workload in sorted(self.workloads, key=lambda item: item.key)
            ],
        }


def release_provenance_from_dict(payload: Mapping[str, Any]) -> ReleaseProvenance:
    """Parse immutable release provenance from a JSON object."""

    return ReleaseProvenance(
        solution=_required_string(payload, "solution"),
        solution_sha256=_required_string(payload, "solution_sha256"),
        environment_fingerprint=_optional_string(payload, "environment_fingerprint"),
        timing_policy=_optional_string(payload, "timing_policy"),
        compiler_build_id=_optional_string(payload, "compiler_build_id"),
        clock_policy=_optional_string(payload, "clock_policy"),
    )


def release_baseline_bundle_from_dict(
    payload: Mapping[str, Any],
) -> ReleaseBaselineBundle:
    """Parse a release-baseline bundle and validate its derived summary."""

    if payload.get("schema_version") != RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported release baseline bundle schema_version")
    provenance = payload.get("provenance")
    workloads = payload.get("workloads")
    if not isinstance(provenance, Mapping) or not isinstance(workloads, list):
        raise ValueError("release baseline bundle requires provenance and workloads")
    bundle = ReleaseBaselineBundle(
        release=_required_string(payload, "release"),
        suite_manifest_ref=_required_string(payload, "suite_manifest_ref"),
        suite_manifest_sha256=_required_string(payload, "suite_manifest_sha256"),
        baseline_artifact_ref=_required_string(payload, "baseline_artifact_ref"),
        baseline_artifact_sha256=_required_string(payload, "baseline_artifact_sha256"),
        provenance=release_provenance_from_dict(provenance),
        workloads=tuple(_workload_from_dict(item) for item in workloads),
        latency_tolerance_rel=float(payload.get("latency_tolerance_rel", 0)),
    )
    if payload.get("summary") != bundle.summary:
        raise ValueError("release baseline bundle summary does not match workloads")
    return bundle


def load_release_baseline_bundle(path: Path) -> ReleaseBaselineBundle:
    """Load and validate a release baseline bundle JSON file."""

    return release_baseline_bundle_from_dict(json.loads(path.read_text()))


def write_release_baseline_bundle(bundle: ReleaseBaselineBundle, path: Path) -> None:
    """Atomically write deterministic release baseline JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    temporary_path.replace(path)


def _workload_from_dict(payload: object) -> ReleaseBaselineWorkload:
    if not isinstance(payload, Mapping):
        raise ValueError("release baseline workload must be an object")
    blockers = payload.get("blocker_reason_codes")
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) for item in blockers
    ):
        raise ValueError("release baseline workload requires blocker_reason_codes list")
    latency = payload.get("latency_ms")
    if latency is not None and not isinstance(latency, (int, float)):
        raise ValueError(
            "release baseline workload latency_ms must be a number or null"
        )
    return ReleaseBaselineWorkload(
        definition=_required_string(payload, "definition"),
        workload_uuid=_required_string(payload, "workload_uuid"),
        classification=_required_string(payload, "classification"),
        latency_ms=float(latency) if latency is not None else None,
        blocker_reason_codes=tuple(blockers),
        trace_ref=_optional_string(payload, "trace_ref"),
        trace_sha256=_optional_string(payload, "trace_sha256"),
        bound_ref=_optional_string(payload, "bound_ref"),
        bound_sha256=_optional_string(payload, "bound_sha256"),
        hardware_model_ref=_optional_string(payload, "hardware_model_ref"),
        hardware_model_sha256=_optional_string(payload, "hardware_model_sha256"),
    )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when present")
    return value
