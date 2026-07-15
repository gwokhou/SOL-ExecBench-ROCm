# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Checksummed shape-aware roofline evidence.

This is deliberately evidence *about* an envelope, rather than a source of a
new latency estimate.  Its purpose is to make the authority gate reject a
hand-written ``shape_aware_roofline`` declaration that has no stable samples,
no launch/occupancy observations, or no binding to the authority slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, cast

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.hardware_calibration.statistics import (
    MINIMUM_SAMPLE_COUNT,
    select_conservative_value,
)


SHAPE_AWARE_ROOFLINE_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline.v1"
SHAPE_AWARE_ROOFLINE_RAW_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline_raw.v1"
_DIMENSIONS = ("shape", "layout", "launch", "occupancy")


def has_measured_occupancy(counters: Mapping[str, object]) -> bool:
    """Return whether raw counters contain a real occupancy observation.

    Dispatch counts and active-cycle counters are useful provenance, but they
    cannot by themselves establish time-integrated wave occupancy.  Keep this
    predicate shared by the collector and artifact validator so a hand-written
    raw JSON cannot promote ``SQ_WAVES`` to occupancy evidence.
    """

    def positive(name: str) -> bool:
        value = counters.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
        )

    return (
        positive("MeanOccupancyPerActiveCU")
        or positive("OccupancyPercent")
        or (positive("SQ_WAVE_CYCLES") and positive("SQ_BUSY_CYCLES"))
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _checksum(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from exc
    return value


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _positive_int_mapping(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    normalized: dict[str, int] = {}
    for key, item in value.items():
        normalized[_non_empty(key, f"{field} key")] = _positive_int(
            item, f"{field}.{key}"
        )
    return normalized


def _validate_patch_provenance(raw: Mapping[str, Any], raw_path: Path) -> None:
    provenance = raw["patch_provenance"]
    expected = {
        "patch_id",
        "wrapper",
        "real_rocprofv3",
        "amd_smi",
        "runtime",
        "performance_levels",
        "clock_lease",
        "raw_profiler_csvs",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected:
        raise ValueError("shape-aware patch provenance has invalid fields")
    _non_empty(provenance["patch_id"], "patch provenance ID")
    for field, version in (
        ("wrapper", False),
        ("real_rocprofv3", True),
        ("amd_smi", True),
    ):
        value = provenance[field]
        if not isinstance(value, Mapping):
            raise ValueError(f"shape-aware patch provenance {field} is invalid")
        path = value.get("path")
        checksum = value.get("sha256")
        _checksum(checksum, f"patch provenance {field} checksum")
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError(f"shape-aware patch provenance {field} path is invalid")
        if sha256_file(Path(path)) != checksum:
            raise ValueError(f"shape-aware patch provenance {field} checksum mismatch")
        if version:
            _non_empty(value.get("version"), f"patch provenance {field} version")
    runtime = provenance["runtime"]
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("driver_and_gpu_static_sha256"), str
    ):
        raise ValueError("shape-aware patch provenance runtime is invalid")
    levels = provenance["performance_levels"]
    if not isinstance(levels, Mapping) or set(levels) != {"pre", "during", "post"}:
        raise ValueError("shape-aware patch provenance levels are invalid")
    if "STABLE_PEAK" not in _non_empty(levels["during"], "patch during level"):
        raise ValueError("shape-aware patch provenance lacks STABLE_PEAK")
    lease = provenance["clock_lease"]
    if not isinstance(lease, Mapping) or lease.get("reset_verified") is not True:
        raise ValueError("shape-aware patch provenance reset is unverified")
    if provenance["raw_profiler_csvs"] != raw["trace_files"]:
        raise ValueError("shape-aware patch provenance CSV binding mismatch")


def validate_shape_aware_raw_evidence(
    case: "ShapeAwareRooflineCase", raw_path: Path, *, architecture: str | None = None
) -> None:
    """Verify a case against a collector-produced raw evidence envelope.

    This deliberately runs only on the authority build path.  A raw file must
    bind the exact workload, tensor layout, samples, dispatch geometry and the
    profiler/provider files it summarizes; an arbitrary checksummed text file
    is not sufficient evidence.
    """
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("shape-aware raw evidence is not valid JSON") from exc
    expected = {
        "schema_version",
        "created_at",
        "architecture",
        "definition",
        "workload_uuid",
        "problem_id",
        "profile_keys",
        "shape",
        "layout",
        "tensor_layouts",
        "samples_ms",
        "provider_evidence_ref",
        "provider_evidence_sha256",
        "trace_files",
        "launch",
        "kernel_resources",
        "occupancy_counters",
        "representative_dispatch_duration_ns",
        "occupancy_status",
        "payload_sha256",
    }
    if not isinstance(raw, dict) or set(raw) not in (
        expected,
        expected | {"patch_provenance"},
    ):
        raise ValueError("shape-aware raw evidence has invalid fields")
    if raw["schema_version"] != SHAPE_AWARE_ROOFLINE_RAW_SCHEMA_VERSION:
        raise ValueError("shape-aware raw evidence schema is unsupported")
    if architecture is not None and raw["architecture"] != architecture:
        raise ValueError("shape-aware raw evidence architecture does not match case")
    unsigned = {key: value for key, value in raw.items() if key != "payload_sha256"}
    if raw["payload_sha256"] != _digest(unsigned):
        raise ValueError("shape-aware raw evidence checksum mismatch")
    if (
        not isinstance(raw["profile_keys"], list)
        or case.profile_key not in raw["profile_keys"]
    ):
        raise ValueError("shape-aware raw evidence does not cover case profile")
    if len(case.covered_workloads) != 1:
        raise ValueError("shape-aware raw evidence requires one workload per case")
    definition, workload_uuid, problem_id = case.covered_workloads[0]
    if (raw["definition"], raw["workload_uuid"], raw["problem_id"]) != (
        definition,
        workload_uuid,
        problem_id,
    ):
        raise ValueError("shape-aware raw evidence workload does not match case")
    if raw["shape"] != list(case.shape) or raw["layout"] != case.layout:
        raise ValueError("shape-aware raw evidence shape/layout does not match case")
    if raw["samples_ms"] != list(case.samples_ms):
        raise ValueError("shape-aware raw evidence samples do not match case")
    if raw["launch"] != dict(case.launch):
        raise ValueError("shape-aware raw evidence launch does not match case")
    if raw["occupancy_status"] != "measured":
        raise ValueError("shape-aware raw evidence lacks measured occupancy")
    counters = raw["occupancy_counters"]
    if not isinstance(counters, dict) or not has_measured_occupancy(counters):
        raise ValueError("shape-aware raw evidence occupancy counters are invalid")
    _bound_raw_file(
        raw_path,
        raw["provider_evidence_ref"],
        raw["provider_evidence_sha256"],
        "provider evidence",
    )
    trace_files = raw["trace_files"]
    if not isinstance(trace_files, list) or not trace_files:
        raise ValueError("shape-aware raw evidence lacks profiler trace files")
    for trace in trace_files:
        if not isinstance(trace, dict) or set(trace) != {"path", "sha256"}:
            raise ValueError("shape-aware raw evidence has invalid profiler trace")
        _bound_raw_file(raw_path, trace["path"], trace["sha256"], "profiler trace")
    if "patch_provenance" in raw:
        _validate_patch_provenance(raw, raw_path)


def _bound_raw_file(
    raw_path: Path, reference: object, checksum: object, label: str
) -> None:
    _checksum(checksum, f"{label} checksum")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"shape-aware raw evidence {label} reference is invalid")
    path = Path(reference)
    if not path.is_absolute():
        path = raw_path.parent / path
    if not path.is_file() or sha256_file(path) != checksum:
        raise ValueError(f"shape-aware raw evidence {label} checksum mismatch")


@dataclass(frozen=True)
class ShapeAwareRooflineCase:
    """One raw, repeatable shape/layout/launch/occupancy probe."""

    case_id: str
    profile_key: str
    shape: tuple[int, ...]
    layout: str
    launch: Mapping[str, int]
    occupancy: Mapping[str, int]
    warmup_iterations: int
    samples_ms: tuple[float, ...]
    covered_workloads: tuple[tuple[str, str, str], ...]
    raw_evidence_ref: str
    raw_evidence_sha256: str

    def __post_init__(self) -> None:
        _non_empty(self.case_id, "case_id")
        _non_empty(self.profile_key, "profile_key")
        if not self.shape:
            raise ValueError("shape must contain at least one positive dimension")
        if any(not isinstance(value, int) or value <= 0 for value in self.shape):
            raise ValueError("shape must contain positive integer dimensions")
        _non_empty(self.layout, "layout")
        for field, mapping in (("launch", self.launch), ("occupancy", self.occupancy)):
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError(f"{field} must be a non-empty object")
            for key, value in mapping.items():
                _non_empty(key, f"{field} key")
                _positive_int(value, f"{field}.{key}")
        _positive_int(self.warmup_iterations, "warmup_iterations")
        if len(self.samples_ms) < MINIMUM_SAMPLE_COUNT:
            raise ValueError(
                f"samples_ms requires at least {MINIMUM_SAMPLE_COUNT} samples"
            )
        samples = tuple(
            _positive_float(value, f"samples_ms[{index}]")
            for index, value in enumerate(self.samples_ms)
        )
        # Stable latency is required just as it is for the scalar probes.  The
        # selection is intentionally not used as a performance ceiling.
        select_conservative_value(tuple(1.0 / value for value in samples))
        if not self.covered_workloads:
            raise ValueError("shape-aware case must cover at least one workload")
        for definition, workload_uuid, problem_id in self.covered_workloads:
            _non_empty(definition, "covered workload definition")
            _non_empty(workload_uuid, "covered workload UUID")
            _non_empty(problem_id, "covered workload problem ID")
        _non_empty(self.raw_evidence_ref, "raw_evidence_ref")
        _checksum(self.raw_evidence_sha256, "raw_evidence_sha256")
        object.__setattr__(self, "samples_ms", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "profile_key": self.profile_key,
            "shape": list(self.shape),
            "layout": self.layout,
            "launch": dict(self.launch),
            "occupancy": dict(self.occupancy),
            "warmup_iterations": self.warmup_iterations,
            "samples_ms": list(self.samples_ms),
            "covered_workloads": [
                {
                    "definition": definition,
                    "workload_uuid": workload_uuid,
                    "problem_id": problem_id,
                }
                for definition, workload_uuid, problem_id in self.covered_workloads
            ],
            "raw_evidence_ref": self.raw_evidence_ref,
            "raw_evidence_sha256": self.raw_evidence_sha256,
        }


@dataclass(frozen=True)
class ShapeAwareRooflineArtifact:
    """An envelope sample set bound to concrete authority inputs."""

    generated_at: str
    architecture: str
    calibration_sha256s: tuple[str, str]
    requirements_sha256: str
    authority_coverage_sha256: str
    plan_payload_sha256: str
    bucketing_dimensions: tuple[str, ...]
    cases: tuple[ShapeAwareRooflineCase, ...]
    collection_status: str
    validation_status: str
    payload_sha256: str | None = None
    schema_version: str = SHAPE_AWARE_ROOFLINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SHAPE_AWARE_ROOFLINE_SCHEMA_VERSION:
            raise ValueError("unsupported shape-aware roofline schema")
        try:
            datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
        _non_empty(self.architecture, "architecture")
        if len(self.calibration_sha256s) != 2:
            raise ValueError("two independent calibration checksums are required")
        if len(set(self.calibration_sha256s)) != 2:
            raise ValueError("calibration checksums must identify independent runs")
        for index, value in enumerate(self.calibration_sha256s):
            _checksum(value, f"calibration_sha256s[{index}]")
        _checksum(self.requirements_sha256, "requirements_sha256")
        _checksum(self.authority_coverage_sha256, "authority_coverage_sha256")
        _checksum(self.plan_payload_sha256, "plan_payload_sha256")
        if len(self.bucketing_dimensions) != len(_DIMENSIONS) or set(
            self.bucketing_dimensions
        ) != set(_DIMENSIONS):
            raise ValueError(
                "bucketing_dimensions must exactly cover shape, layout, launch, and occupancy"
            )
        if (
            self.collection_status != "collected"
            or self.validation_status != "validated"
        ):
            raise ValueError(
                "shape-aware roofline evidence must be collected and validated"
            )
        if not self.cases:
            raise ValueError("shape-aware roofline evidence requires cases")
        identifiers = [case.case_id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("shape-aware roofline case IDs must be unique")
        expected = _digest(self._payload_dict())
        if self.payload_sha256 is not None and self.payload_sha256 != expected:
            raise ValueError("shape-aware roofline payload checksum does not match")
        object.__setattr__(self, "payload_sha256", expected)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "architecture": self.architecture,
            "calibration_sha256s": list(self.calibration_sha256s),
            "requirements_sha256": self.requirements_sha256,
            "authority_coverage_sha256": self.authority_coverage_sha256,
            "plan_payload_sha256": self.plan_payload_sha256,
            "bucketing_dimensions": list(self.bucketing_dimensions),
            "cases": [case.to_dict() for case in self.cases],
            "collection_status": self.collection_status,
            "validation_status": self.validation_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "payload_sha256": self.payload_sha256}

    @property
    def profile_keys(self) -> frozenset[str]:
        return frozenset(case.profile_key for case in self.cases)


def shape_aware_roofline_from_dict(
    payload: Mapping[str, Any],
) -> ShapeAwareRooflineArtifact:
    """Strictly parse one checksummed shape-aware roofline artifact."""
    expected = {
        "schema_version",
        "generated_at",
        "architecture",
        "calibration_sha256s",
        "requirements_sha256",
        "authority_coverage_sha256",
        "plan_payload_sha256",
        "bucketing_dimensions",
        "cases",
        "collection_status",
        "validation_status",
        "payload_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("shape-aware roofline artifact has invalid fields")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("shape-aware roofline cases must be a list")
    cases: list[ShapeAwareRooflineCase] = []
    case_fields = {
        "case_id",
        "profile_key",
        "shape",
        "layout",
        "launch",
        "occupancy",
        "warmup_iterations",
        "samples_ms",
        "covered_workloads",
        "raw_evidence_ref",
        "raw_evidence_sha256",
    }
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping) or set(raw) != case_fields:
            raise ValueError(f"shape-aware roofline case {index} has invalid fields")
        case_payload = cast(Mapping[str, object], raw)
        raw_shape = case_payload["shape"]
        raw_samples = case_payload["samples_ms"]
        if not isinstance(raw_shape, list) or not isinstance(raw_samples, list):
            raise ValueError(
                f"shape-aware roofline case {index} shape and samples_ms must be lists"
            )
        covered_workloads = case_payload["covered_workloads"]
        if not isinstance(covered_workloads, list):
            raise ValueError(
                f"shape-aware roofline case {index} covered_workloads must be a list"
            )
        workload_keys: list[tuple[str, str, str]] = []
        for workload in covered_workloads:
            expected_workload = {"definition", "workload_uuid", "problem_id"}
            if not isinstance(workload, Mapping) or set(workload) != expected_workload:
                raise ValueError(
                    f"shape-aware roofline case {index} has invalid covered workload"
                )
            workload_payload = cast(Mapping[str, object], workload)
            workload_keys.append(
                (
                    _non_empty(
                        workload_payload["definition"],
                        "covered workload definition",
                    ),
                    _non_empty(
                        workload_payload["workload_uuid"], "covered workload UUID"
                    ),
                    _non_empty(
                        workload_payload["problem_id"], "covered workload problem ID"
                    ),
                )
            )
        cases.append(
            ShapeAwareRooflineCase(
                case_id=_non_empty(case_payload["case_id"], "case_id"),
                profile_key=_non_empty(case_payload["profile_key"], "profile_key"),
                shape=tuple(
                    _positive_int(value, f"shape[{position}]")
                    for position, value in enumerate(raw_shape)
                ),
                layout=_non_empty(case_payload["layout"], "layout"),
                launch=_positive_int_mapping(case_payload["launch"], "launch"),
                occupancy=_positive_int_mapping(case_payload["occupancy"], "occupancy"),
                warmup_iterations=_positive_int(
                    case_payload["warmup_iterations"], "warmup_iterations"
                ),
                samples_ms=tuple(
                    _positive_float(value, f"samples_ms[{position}]")
                    for position, value in enumerate(raw_samples)
                ),
                covered_workloads=tuple(workload_keys),
                raw_evidence_ref=_non_empty(
                    case_payload["raw_evidence_ref"], "raw_evidence_ref"
                ),
                raw_evidence_sha256=_checksum(
                    case_payload["raw_evidence_sha256"], "raw_evidence_sha256"
                ),
            )
        )
    raw_calibrations = payload["calibration_sha256s"]
    dimensions = payload["bucketing_dimensions"]
    if not isinstance(raw_calibrations, list) or not isinstance(dimensions, list):
        raise ValueError(
            "shape-aware roofline checksum and dimension fields must be lists"
        )
    if len(raw_calibrations) != 2:
        raise ValueError("two independent calibration checksums are required")
    return ShapeAwareRooflineArtifact(
        schema_version=_non_empty(payload["schema_version"], "schema_version"),
        generated_at=_non_empty(payload["generated_at"], "generated_at"),
        architecture=_non_empty(payload["architecture"], "architecture"),
        calibration_sha256s=(
            _checksum(raw_calibrations[0], "calibration_sha256s[0]"),
            _checksum(raw_calibrations[1], "calibration_sha256s[1]"),
        ),
        requirements_sha256=_checksum(
            payload["requirements_sha256"], "requirements_sha256"
        ),
        authority_coverage_sha256=_checksum(
            payload["authority_coverage_sha256"], "authority_coverage_sha256"
        ),
        plan_payload_sha256=_checksum(
            payload["plan_payload_sha256"], "plan_payload_sha256"
        ),
        bucketing_dimensions=tuple(
            _non_empty(item, "bucketing_dimensions") for item in dimensions
        ),
        cases=tuple(cases),
        collection_status=_non_empty(payload["collection_status"], "collection_status"),
        validation_status=_non_empty(payload["validation_status"], "validation_status"),
        payload_sha256=_checksum(payload["payload_sha256"], "payload_sha256"),
    )
