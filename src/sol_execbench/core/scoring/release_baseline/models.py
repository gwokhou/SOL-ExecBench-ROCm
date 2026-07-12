# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, release-scoped baseline evidence contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast


RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION = "sol_execbench.release_baseline_bundle.v1"
RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION = (
    "sol_execbench.release_baseline_verification.v1"
)
CLASSIFICATIONS = ("official", "derived", "blocked")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a 64-character lowercase sha256 digest")


def _require_optional_sha256(value: str | None, name: str) -> None:
    if value is not None:
        _require_sha256(value, name)


def _require_optional_reference(ref: str | None, digest: str | None, name: str) -> None:
    if (ref is None) != (digest is None):
        raise ValueError(f"{name}_ref and {name}_sha256 must be provided together")
    if ref is not None:
        _require_non_empty(ref, f"{name}_ref")
    _require_optional_sha256(digest, f"{name}_sha256")


@dataclass(frozen=True)
class ReleaseProvenance:
    """Fixed optimized-solution identity used to produce a release baseline."""

    solution: str
    solution_sha256: str
    environment_fingerprint: str | None = None
    clock_policy: str | None = None
    compiler_build_id: str | None = None
    timing_policy: str | None = None
    suite_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.solution, "solution")
        _require_sha256(self.solution_sha256, "solution_sha256")
        for name in (
            "environment_fingerprint",
            "clock_policy",
            "compiler_build_id",
            "timing_policy",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(value, name)
        _require_optional_sha256(self.suite_manifest_sha256, "suite_manifest_sha256")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "solution": self.solution,
            "solution_sha256": self.solution_sha256,
            "environment_fingerprint": self.environment_fingerprint,
            "clock_policy": self.clock_policy,
            "compiler_build_id": self.compiler_build_id,
            "timing_policy": self.timing_policy,
            "suite_manifest_sha256": self.suite_manifest_sha256,
        }


@dataclass(frozen=True)
class ReleaseBaselineWorkload:
    """One immutable selected-suite denominator row."""

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
            raise ValueError(
                f"classification must be one of {CLASSIFICATIONS!r}; "
                f"got {self.classification!r}"
            )
        if self.latency_ms is not None and (
            not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0
        ):
            raise ValueError("latency_ms must be positive and finite when present")
        if any(
            not isinstance(code, str) or not code.strip()
            for code in self.blocker_reason_codes
        ):
            raise ValueError("blocker_reason_codes must contain non-empty strings")
        _require_optional_reference(self.trace_ref, self.trace_sha256, "trace")
        _require_optional_reference(self.bound_ref, self.bound_sha256, "bound")
        _require_optional_reference(
            self.hardware_model_ref, self.hardware_model_sha256, "hardware_model"
        )
        if self.classification == "official" and (
            self.trace_ref is None
            or self.bound_ref is None
            or self.hardware_model_ref is None
        ):
            raise ValueError(
                "official workloads require trace, bound, and hardware model evidence"
            )
        if self.classification == "official" and self.latency_ms is None:
            raise ValueError("official workloads require a measured latency_ms")

    @property
    def key(self) -> tuple[str, str]:
        return (self.definition, self.workload_uuid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "classification": self.classification,
            "latency_ms": self.latency_ms,
            "blocker_reason_codes": list(self.blocker_reason_codes),
            "trace_ref": self.trace_ref,
            "trace_sha256": self.trace_sha256,
            "bound_ref": self.bound_ref,
            "bound_sha256": self.bound_sha256,
            "hardware_model_ref": self.hardware_model_ref,
            "hardware_model_sha256": self.hardware_model_sha256,
        }


@dataclass(frozen=True)
class ReleaseBaselineBundle:
    """Versioned public evidence for a complete selected release suite."""

    release: str
    suite_manifest_ref: str
    suite_manifest_sha256: str
    baseline_artifact_ref: str
    baseline_artifact_sha256: str
    provenance: ReleaseProvenance
    workloads: tuple[ReleaseBaselineWorkload, ...]
    latency_tolerance_rel: float
    scope: str = "unspecified"
    schema_version: str = RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.release, "release")
        _require_non_empty(self.suite_manifest_ref, "suite_manifest_ref")
        _require_sha256(self.suite_manifest_sha256, "suite_manifest_sha256")
        _require_non_empty(self.baseline_artifact_ref, "baseline_artifact_ref")
        _require_sha256(self.baseline_artifact_sha256, "baseline_artifact_sha256")
        _require_non_empty(self.scope, "scope")
        if (
            not math.isfinite(self.latency_tolerance_rel)
            or self.latency_tolerance_rel <= 0.0
        ):
            raise ValueError("latency_tolerance_rel must be positive and finite")
        if self.schema_version != RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION!r}"
            )
        keys = [workload.key for workload in self.workloads]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate workload key in release baseline bundle")
        if self.summary["total"] != sum(self.summary[name] for name in CLASSIFICATIONS):
            raise ValueError("summary total must equal classification counts")

    @property
    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.workloads),
            **{
                classification: sum(
                    workload.classification == classification
                    for workload in self.workloads
                )
                for classification in CLASSIFICATIONS
            },
        }

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
            "scope": self.scope,
            "summary": self.summary,
            "workloads": [
                workload.to_dict()
                for workload in sorted(self.workloads, key=lambda row: row.key)
            ],
        }


@dataclass(frozen=True)
class ReleaseBaselineVerification:
    """Immutable reference to a verification run for a published bundle."""

    release: str
    bundle_ref: str
    bundle_sha256: str
    rerun_trace_ref: str
    rerun_trace_sha256: str
    workloads: tuple["ReleaseBaselineVerificationWorkload", ...] = ()
    schema_version: str = RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.release, "release")
        _require_non_empty(self.bundle_ref, "bundle_ref")
        _require_sha256(self.bundle_sha256, "bundle_sha256")
        _require_non_empty(self.rerun_trace_ref, "rerun_trace_ref")
        _require_sha256(self.rerun_trace_sha256, "rerun_trace_sha256")
        if self.schema_version != RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION!r}"
            )
        keys = [workload.key for workload in self.workloads]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate workload key in release baseline verification")
        if self.summary["total"] != sum(self.summary[name] for name in CLASSIFICATIONS):
            raise ValueError(
                "verification summary total must equal classification counts"
            )

    @property
    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.workloads),
            **{
                classification: sum(
                    workload.classification == classification
                    for workload in self.workloads
                )
                for classification in CLASSIFICATIONS
            },
            "passed": sum(workload.passed for workload in self.workloads),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release": self.release,
            "bundle_ref": self.bundle_ref,
            "bundle_sha256": self.bundle_sha256,
            "rerun_trace_ref": self.rerun_trace_ref,
            "rerun_trace_sha256": self.rerun_trace_sha256,
            "summary": self.summary,
            "workloads": [
                workload.to_dict()
                for workload in sorted(self.workloads, key=lambda row: row.key)
            ],
        }


@dataclass(frozen=True)
class ReleaseBaselineVerificationWorkload:
    """One workload's immutable rerun comparison result."""

    definition: str
    workload_uuid: str
    original_classification: str
    classification: str
    baseline_latency_ms: float | None
    rerun_latency_ms: float | None
    latency_delta_rel: float | None
    passed: bool
    blocker_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.definition, "definition")
        _require_non_empty(self.workload_uuid, "workload_uuid")
        if self.original_classification not in CLASSIFICATIONS:
            raise ValueError("original_classification must be a known classification")
        if self.classification not in CLASSIFICATIONS:
            raise ValueError("classification must be a known classification")
        for name in ("baseline_latency_ms", "rerun_latency_ms"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when present")
        if self.latency_delta_rel is not None and (
            not math.isfinite(self.latency_delta_rel) or self.latency_delta_rel < 0.0
        ):
            raise ValueError(
                "latency_delta_rel must be finite and nonnegative when present"
            )
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool")
        if any(
            not isinstance(code, str) or not code.strip()
            for code in self.blocker_reason_codes
        ):
            raise ValueError("blocker_reason_codes must contain non-empty strings")
        if self.passed != (self.classification in ("official", "derived")):
            raise ValueError("passed must match a non-blocked final classification")
        if self.classification != "blocked" and (
            self.baseline_latency_ms is None
            or self.rerun_latency_ms is None
            or self.latency_delta_rel is None
        ):
            raise ValueError(
                "non-blocked verification workloads require baseline, rerun, and delta measurements"
            )

    @property
    def key(self) -> tuple[str, str]:
        return self.definition, self.workload_uuid

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "original_classification": self.original_classification,
            "classification": self.classification,
            "baseline_latency_ms": self.baseline_latency_ms,
            "rerun_latency_ms": self.rerun_latency_ms,
            "latency_delta_rel": self.latency_delta_rel,
            "passed": self.passed,
            "blocker_reason_codes": list(self.blocker_reason_codes),
        }


def release_baseline_bundle_from_dict(
    payload: Mapping[str, Any],
) -> ReleaseBaselineBundle:
    """Parse and validate a release baseline bundle JSON payload."""

    if payload.get("schema_version") != RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "release baseline bundle requires schema_version "
            f"{RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION!r}"
        )
    raw_provenance = payload.get("provenance")
    raw_workloads = payload.get("workloads")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("release baseline bundle requires a provenance object")
    if not isinstance(raw_workloads, list):
        raise ValueError("release baseline bundle requires a workloads list")
    try:
        provenance = ReleaseProvenance(
            solution=str(raw_provenance["solution"]),
            solution_sha256=str(raw_provenance["solution_sha256"]),
            environment_fingerprint=_optional_string(
                raw_provenance.get("environment_fingerprint")
            ),
            clock_policy=_optional_string(raw_provenance.get("clock_policy")),
            compiler_build_id=_optional_string(raw_provenance.get("compiler_build_id")),
            timing_policy=_optional_string(raw_provenance.get("timing_policy")),
            suite_manifest_sha256=_optional_string(
                raw_provenance.get("suite_manifest_sha256")
            ),
        )
        workloads = tuple(
            _workload_from_dict(row, index) for index, row in enumerate(raw_workloads)
        )
        bundle = ReleaseBaselineBundle(
            release=str(payload["release"]),
            suite_manifest_ref=str(payload["suite_manifest_ref"]),
            suite_manifest_sha256=str(payload["suite_manifest_sha256"]),
            baseline_artifact_ref=str(payload["baseline_artifact_ref"]),
            baseline_artifact_sha256=str(payload["baseline_artifact_sha256"]),
            provenance=provenance,
            workloads=workloads,
            latency_tolerance_rel=float(payload["latency_tolerance_rel"]),
            scope=str(payload.get("scope", "unspecified")),
            schema_version=str(payload["schema_version"]),
        )
    except KeyError as exc:
        raise ValueError(f"release baseline bundle missing {exc.args[0]}") from exc
    if payload.get("summary") != bundle.summary:
        raise ValueError("release baseline bundle summary does not match workloads")
    return bundle


def _workload_from_dict(raw: Any, index: int) -> ReleaseBaselineWorkload:
    if not isinstance(raw, Mapping):
        raise ValueError(f"release baseline workload {index} must be an object")
    blocker_reason_codes = raw.get("blocker_reason_codes")
    if not isinstance(blocker_reason_codes, list):
        raise ValueError(
            f"release baseline workload {index} requires blocker_reason_codes"
        )
    try:
        return ReleaseBaselineWorkload(
            definition=str(raw["definition"]),
            workload_uuid=str(raw["workload_uuid"]),
            classification=str(raw["classification"]),
            latency_ms=(
                float(raw["latency_ms"]) if raw.get("latency_ms") is not None else None
            ),
            blocker_reason_codes=tuple(str(code) for code in blocker_reason_codes),
            trace_ref=_optional_string(raw.get("trace_ref")),
            trace_sha256=_optional_string(raw.get("trace_sha256")),
            bound_ref=_optional_string(raw.get("bound_ref")),
            bound_sha256=_optional_string(raw.get("bound_sha256")),
            hardware_model_ref=_optional_string(raw.get("hardware_model_ref")),
            hardware_model_sha256=_optional_string(raw.get("hardware_model_sha256")),
        )
    except KeyError as exc:
        raise ValueError(
            f"release baseline workload {index} missing {exc.args[0]}"
        ) from exc


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def write_release_baseline_bundle(bundle: ReleaseBaselineBundle, path: Path) -> None:
    """Write deterministic JSON for a release baseline bundle."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_release_baseline_bundle(path: Path) -> ReleaseBaselineBundle:
    """Load a validated release baseline bundle from JSON."""

    return release_baseline_bundle_from_dict(
        cast(Mapping[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    )


def release_baseline_verification_from_dict(
    payload: Mapping[str, Any],
) -> ReleaseBaselineVerification:
    """Parse and validate a release baseline verification JSON payload."""

    if payload.get("schema_version") != RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION:
        raise ValueError(
            "release baseline verification has an unsupported schema_version"
        )
    raw_workloads = payload.get("workloads", [])
    if not isinstance(raw_workloads, list):
        raise ValueError("release baseline verification requires a workloads list")
    try:
        verification = ReleaseBaselineVerification(
            release=str(payload["release"]),
            bundle_ref=str(payload["bundle_ref"]),
            bundle_sha256=str(payload["bundle_sha256"]),
            rerun_trace_ref=str(payload["rerun_trace_ref"]),
            rerun_trace_sha256=str(payload["rerun_trace_sha256"]),
            workloads=tuple(
                _verification_workload_from_dict(row, index)
                for index, row in enumerate(raw_workloads)
            ),
            schema_version=str(payload["schema_version"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"release baseline verification missing {exc.args[0]}"
        ) from exc
    if "summary" in payload and payload["summary"] != verification.summary:
        raise ValueError(
            "release baseline verification summary does not match workloads"
        )
    return verification


def _verification_workload_from_dict(
    raw: Any, index: int
) -> ReleaseBaselineVerificationWorkload:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"release baseline verification workload {index} must be an object"
        )
    codes = raw.get("blocker_reason_codes")
    if not isinstance(codes, list):
        raise ValueError(
            f"release baseline verification workload {index} requires blocker_reason_codes"
        )
    try:
        return ReleaseBaselineVerificationWorkload(
            definition=str(raw["definition"]),
            workload_uuid=str(raw["workload_uuid"]),
            original_classification=str(raw["original_classification"]),
            classification=str(raw["classification"]),
            baseline_latency_ms=(
                float(raw["baseline_latency_ms"])
                if raw.get("baseline_latency_ms") is not None
                else None
            ),
            rerun_latency_ms=(
                float(raw["rerun_latency_ms"])
                if raw.get("rerun_latency_ms") is not None
                else None
            ),
            latency_delta_rel=(
                float(raw["latency_delta_rel"])
                if raw.get("latency_delta_rel") is not None
                else None
            ),
            passed=raw["passed"],
            blocker_reason_codes=tuple(str(code) for code in codes),
        )
    except KeyError as exc:
        raise ValueError(
            f"release baseline verification workload {index} missing {exc.args[0]}"
        ) from exc


def write_release_baseline_verification(
    verification: ReleaseBaselineVerification, path: Path
) -> None:
    """Write deterministic JSON for a release baseline verification report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verification.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
