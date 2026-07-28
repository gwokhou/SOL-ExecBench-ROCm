# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Normalized AMD ROCm architecture profiles used by SOL roofline calculations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml

from solar import schema_versions
from solar.common.constants import normalize_dtype
from solar.rocm.audit_validation import (
    audit_mapping,
    audit_nonnegative_int,
    audit_sha256,
    audit_string_set,
)

# The hash-bound calibration script imports this legacy path. Keep the explicit
# compatibility binding until a new calibration artifact is published.
RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION = (
    schema_versions.RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION
)
RESOURCE_PEAK_TIMING_PROFILE = "official"
UNTHROTTLED_RESOURCE_PEAK_SCOPE = "unthrottled_resource_peak"
INSTRUCTION_RUNTIME_AUDIT_SCOPE = "instruction_and_runtime_corroboration_only"
_MAX_AUDIT_BYTES = 2 * 1024 * 1024

_PRECISION_ALIASES = {
    "float32": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float8": "fp8",
}

_VENDOR_SPECIFIC_DTYPES = frozenset(
    {
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float4_e2m1",
        "float4_e2m1fn_x2",
    },
)


def _packaged_profile_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "arch"
        / f"{name}.yaml"
    )


def resource_peak_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the resource-audit digest with ``payload_sha256`` omitted."""
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_resource_peak_audit(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema_version: str,
    expected_timing_profile: str,
    expected_clocks_locked: bool,
    expected_unthrottled: bool,
    expected_gfx_target: str,
    expected_precisions: tuple[str, ...],
    expected_resource_modes: tuple[str, ...],
    expected_instruction_checks: tuple[str, ...],
) -> dict[str, Any]:
    """Load and validate one content-addressed resource-peak audit artifact."""
    if not path.is_file():
        raise ValueError(f"architecture audit evidence file is missing: {path}")
    if path.stat().st_size > _MAX_AUDIT_BYTES:
        raise ValueError("architecture audit evidence exceeds the size limit")
    with path.open("rb") as audit:
        observed_sha256 = hashlib.file_digest(audit, "sha256").hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("architecture audit evidence identity mismatch")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "architecture audit evidence is not valid JSON",
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("architecture audit evidence must be a JSON object")
    payload = cast(dict[str, Any], raw)
    if payload.get("payload_sha256") != resource_peak_payload_sha256(payload):
        raise ValueError(
            "architecture audit evidence payload checksum mismatch",
        )
    _verify_audit_contract(
        payload,
        expected_schema_version=expected_schema_version,
        expected_timing_profile=expected_timing_profile,
        expected_clocks_locked=expected_clocks_locked,
        expected_gfx_target=expected_gfx_target,
    )
    from solar.rocm.resource_peak_measurements import (
        verify_resource_peak_measurements,
    )

    covered_precisions, covered_resources, raw_measurements = (
        verify_resource_peak_measurements(
            payload.get("experiment_protocol"),
            payload.get("measurements"),
            require_unthrottled=expected_unthrottled,
        )
    )
    measurements = cast(dict[str, Mapping[str, Any]], raw_measurements)
    instruction_presence, isa_spec_sha256 = _verify_audit_isa_spec(
        payload.get("isa_spec_evidence"),
        expected_gfx_target=expected_gfx_target,
    )
    _verify_audit_coverage(
        payload.get("calibration_coverage"),
        expected_precisions=expected_precisions,
        expected_resource_modes=expected_resource_modes,
        covered_precisions=covered_precisions,
        covered_resource_modes=covered_resources,
    )
    _verify_audit_instruction_checks(
        payload.get("instruction_validation"),
        expected_checks=expected_instruction_checks,
        measurements=measurements,
        instruction_presence=instruction_presence,
        isa_spec_sha256=isa_spec_sha256,
    )
    return payload


def _verify_audit_contract(
    payload: Mapping[str, Any],
    *,
    expected_schema_version: str,
    expected_timing_profile: str,
    expected_clocks_locked: bool,
    expected_gfx_target: str,
) -> None:
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError("architecture audit evidence schema mismatch")
    if payload.get("timing_profile") != expected_timing_profile:
        raise ValueError("architecture audit evidence timing profile mismatch")
    device = audit_mapping(payload.get("device"), "device")
    if device.get("gfx_target") != expected_gfx_target:
        raise ValueError("architecture audit evidence gfx target mismatch")
    if not str(device.get("device_name", "")).strip():
        raise ValueError("architecture audit evidence lacks a GPU identity")
    clock_setup = audit_mapping(payload.get("clock_setup"), "clock_setup")
    if bool(clock_setup.get("clock_locked_verified")) != expected_clocks_locked:
        raise ValueError(
            "architecture audit evidence clock-lock state mismatch",
        )


def _verify_audit_isa_spec(
    value: object,
    *,
    expected_gfx_target: str,
) -> tuple[Mapping[str, bool], str]:
    evidence = audit_mapping(value, "isa_spec_evidence")
    if evidence.get("architecture") != expected_gfx_target:
        raise ValueError("architecture audit ISA evidence target mismatch")
    provenance = audit_mapping(evidence.get("provenance"), "ISA provenance")
    spec_sha256 = str(provenance.get("spec_sha256", ""))
    if (
        provenance.get("architecture") != expected_gfx_target
        or not spec_sha256.strip()
        or not str(provenance.get("release", "")).strip()
    ):
        raise ValueError("architecture audit ISA provenance is incomplete")
    raw_presence = audit_mapping(
        evidence.get("instruction_presence"),
        "instruction presence",
    )
    if not raw_presence or any(
        not isinstance(value, bool) for value in raw_presence.values()
    ):
        raise ValueError(
            "architecture audit instruction presence must be boolean",
        )
    return cast(Mapping[str, bool], raw_presence), spec_sha256


def _verify_audit_coverage(
    value: object,
    *,
    expected_precisions: tuple[str, ...],
    expected_resource_modes: tuple[str, ...],
    covered_precisions: set[str],
    covered_resource_modes: set[str],
) -> None:
    coverage = audit_mapping(value, "calibration_coverage")
    if coverage.get("status") != "passed":
        raise ValueError("architecture audit calibration coverage did not pass")
    comparisons = (
        ("required_precisions", set(expected_precisions)),
        ("covered_precisions", covered_precisions),
        ("required_resource_modes", set(expected_resource_modes)),
        ("covered_resource_modes", covered_resource_modes),
    )
    for field_name, expected in comparisons:
        if audit_string_set(coverage.get(field_name), field_name) != expected:
            raise ValueError(f"architecture audit {field_name} mismatch")
    if covered_precisions != set(expected_precisions):
        raise ValueError(
            "architecture audit precision calibration coverage mismatch",
        )
    if covered_resource_modes != set(expected_resource_modes):
        raise ValueError(
            "architecture audit resource calibration coverage mismatch",
        )


def _verify_audit_instruction_checks(
    value: object,
    *,
    expected_checks: tuple[str, ...],
    measurements: Mapping[str, Mapping[str, Any]],
    instruction_presence: Mapping[str, bool],
    isa_spec_sha256: str,
) -> None:
    validation = audit_mapping(value, "instruction_validation")
    if validation.get("status") != "passed":
        raise ValueError(
            "architecture audit instruction validation did not pass",
        )
    required = audit_string_set(
        validation.get("required_checks"),
        "instruction required_checks",
    )
    if required != set(expected_checks):
        raise ValueError(
            "architecture audit required instruction checks mismatch",
        )
    checks = audit_mapping(validation.get("checks"), "instruction checks")
    if set(checks) != required:
        raise ValueError("architecture audit instruction check set mismatch")
    for name in sorted(required):
        _verify_audit_instruction_check(
            audit_mapping(checks[name], f"instruction check {name}"),
            measurements=measurements,
            instruction_presence=instruction_presence,
            isa_spec_sha256=isa_spec_sha256,
        )


def _verify_audit_instruction_check(
    check: Mapping[str, Any],
    *,
    measurements: Mapping[str, Mapping[str, Any]],
    instruction_presence: Mapping[str, bool],
    isa_spec_sha256: str,
) -> None:
    if (
        check.get("status") != "passed"
        or check.get("runtime_probe_passed") is not True
    ):
        raise ValueError("architecture audit instruction check did not pass")
    probe = str(check.get("probe", ""))
    measurement = measurements.get(probe)
    if measurement is None:
        raise ValueError(
            "architecture audit instruction check references unknown probe",
        )
    compiler_isa = audit_mapping(
        measurement.get("compiler_isa"),
        "measurement compiler_isa",
    )
    compiler_provenance = audit_mapping(
        compiler_isa.get("spec_provenance"),
        "compiler ISA provenance",
    )
    if compiler_provenance.get("spec_sha256") != isa_spec_sha256:
        raise ValueError(
            "architecture audit compiler ISA specification mismatch",
        )
    counts = audit_mapping(
        compiler_isa.get("matched_instruction_counts"),
        "matched instruction counts",
    )
    instructions = audit_string_set(check.get("instructions"), "instructions")
    if not instructions.issubset(instruction_presence):
        raise ValueError(
            "architecture audit instruction is absent from ISA evidence",
        )
    declared = any(instruction_presence[name] for name in instructions)
    if declared is not check.get("isa_declared"):
        raise ValueError("architecture audit ISA declaration mismatch")
    emitted = sum(
        audit_nonnegative_int(counts.get(name), "instruction count")
        for name in instructions
    )
    claimed_emitted = audit_nonnegative_int(
        check.get("compiler_emitted_count"),
        "compiler emitted count",
    )
    if emitted != claimed_emitted:
        raise ValueError(
            "architecture audit compiler instruction count mismatch",
        )
    _verify_instruction_expectation(
        check,
        instruction_presence,
        counts,
        instructions,
        emitted,
    )


def _verify_instruction_expectation(
    check: Mapping[str, Any],
    instruction_presence: Mapping[str, bool],
    counts: Mapping[str, Any],
    instructions: set[str],
    emitted: int,
) -> None:
    expectation = check.get("expectation")
    if expectation == "native":
        if (
            check.get("isa_declared") is not True
            or emitted <= 0
            or check.get("native_instruction_usable") is not True
        ):
            raise ValueError(
                "architecture audit native instruction evidence mismatch",
            )
        return
    if expectation != "fallback":
        raise ValueError(
            "architecture audit instruction expectation is invalid",
        )
    fallback_names = audit_string_set(
        check.get("fallback_instructions"),
        "fallback instructions",
    )
    if not fallback_names.issubset(instruction_presence) or not any(
        instruction_presence[name] for name in fallback_names
    ):
        raise ValueError(
            "architecture audit fallback is absent from ISA evidence",
        )
    fallback_count = sum(
        audit_nonnegative_int(counts.get(name), "fallback instruction count")
        for name in fallback_names
    )
    if (
        check.get("isa_declared") is not False
        or emitted != 0
        or fallback_count <= 0
        or fallback_count
        != audit_nonnegative_int(
            check.get("fallback_emitted_count"),
            "fallback emitted count",
        )
        or check.get("native_instruction_usable") is not False
    ):
        raise ValueError(
            "architecture audit fallback instruction evidence mismatch",
        )


def _validate_audit_evidence_config(evidence: Mapping[str, Any]) -> None:
    status = str(evidence.get("status", ""))
    if status not in {"verified", "unavailable"}:
        raise ValueError(
            "audit_evidence.status must be verified or unavailable",
        )
    evidence_sha = str(evidence.get("sha256", ""))
    if evidence_sha:
        try:
            audit_sha256(evidence_sha, "audit_evidence.sha256")
        except ValueError as exc:
            raise ValueError(
                "audit_evidence.sha256 must be a lowercase SHA-256",
            ) from exc
    if status != "verified":
        return
    if not evidence_sha:
        raise ValueError("verified audit evidence requires a SHA-256")
    required_fields = {
        "path",
        "required_schema_version",
        "required_timing_profile",
        "required_clocks_locked",
        "required_unthrottled",
        "evidence_scope",
        "gfx_target",
        "required_instruction_checks",
    }
    missing_fields = sorted(required_fields - set(evidence))
    if missing_fields:
        raise ValueError(
            f"verified audit evidence lacks required fields: {missing_fields}",
        )
    required_unthrottled = evidence.get("required_unthrottled")
    evidence_scope = evidence.get("evidence_scope")
    if not isinstance(required_unthrottled, bool):
        raise ValueError(
            "verified audit evidence requires a boolean throttle policy",
        )
    expected_scope = (
        UNTHROTTLED_RESOURCE_PEAK_SCOPE
        if required_unthrottled
        else INSTRUCTION_RUNTIME_AUDIT_SCOPE
    )
    if evidence_scope != expected_scope:
        raise ValueError(
            "verified audit evidence scope contradicts throttle policy",
        )
    checks = evidence.get("required_instruction_checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(item, str) or not item.strip() for item in checks)
        or len(set(checks)) != len(checks)
    ):
        raise ValueError(
            "verified audit evidence requires unique instruction check names",
        )


@dataclass(frozen=True)
class MemoryLevel:
    """One explicitly sourced AMD memory level; unknown values stay unknown."""

    name: str
    scope: str
    capacity_bytes: int | None
    bandwidth_bytes_per_second: float | None = None
    source: str | None = None

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> MemoryLevel:
        """Build a memory level from serialized profile data."""
        capacity = data.get("capacity_bytes")
        bandwidth = data.get("bandwidth_bytes_per_second")
        return cls(
            name=str(data.get("name", "")),
            scope=str(data.get("scope", "")),
            capacity_bytes=int(capacity) if capacity is not None else None,
            bandwidth_bytes_per_second=(
                float(bandwidth) if bandwidth is not None else None
            ),
            source=str(data.get("source", "")) or None,
        )


def _load_profile_data(
    value: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}, None
    path = Path(value)
    if not path.exists():
        root = Path(__file__).resolve().parents[2]
        path = root / "configs" / "arch" / f"{value}.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}, str(path)
    resource = resources.files("solar.configs.arch").joinpath(f"{value}.yaml")
    if not resource.is_file():
        raise FileNotFoundError(f"Architecture profile not found: {value}")
    return yaml.safe_load(resource.read_text()) or {}, str(resource)


@dataclass(frozen=True)
class ArchitectureProfile:
    """Normalized AMD hardware limits used by SOL roofline calculations."""

    name: str
    vendor: str
    gfx_target: str
    compute_units: int
    memory_capacity_bytes: int
    memory_bandwidth_bytes_per_second: float
    l2_bytes: int
    last_level_cache_bytes: int
    peak_ops_per_second: dict[str, float] = field(default_factory=dict)
    resource_model_version: str = ""
    resource_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    resource_limit_sources: dict[str, str] = field(default_factory=dict)
    calibration_exempt_modes: dict[str, dict[str, str]] = field(
        default_factory=dict,
    )
    precision_support: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_revision: str = ""
    audit_evidence: dict[str, Any] = field(default_factory=dict)
    precision_aliases: dict[str, str] = field(default_factory=dict)
    clock_hz: float | None = None
    source: str | None = None
    memory_hierarchy: tuple[MemoryLevel, ...] = ()

    @classmethod
    def load(cls, value: str | Path | Mapping[str, Any]) -> ArchitectureProfile:
        """Load a normalized AMD ROCm architecture description."""
        data, source = _load_profile_data(value)
        if "peak_ops_per_second" not in data:
            raise ValueError(
                "ROCm architecture profiles must define normalized "
                "peak_ops_per_second fields",
            )
        profile = cls(
            name=str(data["name"]),
            vendor=str(data.get("vendor", "")),
            gfx_target=str(data.get("gfx_target", "")),
            compute_units=int(data.get("compute_units", 0)),
            memory_capacity_bytes=int(data.get("memory_capacity_bytes", 0)),
            memory_bandwidth_bytes_per_second=float(
                data["memory_bandwidth_bytes_per_second"],
            ),
            l2_bytes=int(data.get("l2_bytes", 0)),
            last_level_cache_bytes=int(data.get("last_level_cache_bytes", 0)),
            peak_ops_per_second={
                str(k).lower(): float(v)
                for k, v in data["peak_ops_per_second"].items()
            },
            resource_model_version=str(data.get("resource_model_version", "")),
            resource_limits={
                str(resource_name).lower(): {
                    str(mode).lower(): float(rate)
                    for mode, rate in (modes or {}).items()
                }
                for resource_name, modes in (
                    data.get("resource_limits") or {}
                ).items()
            },
            resource_limit_sources={
                str(key).lower(): str(value)
                for key, value in (
                    data.get("resource_limit_sources") or {}
                ).items()
            },
            calibration_exempt_modes={
                str(resource_name).lower(): {
                    str(mode).lower(): str(reason)
                    for mode, reason in (modes or {}).items()
                }
                for resource_name, modes in (
                    data.get("calibration_exempt_modes") or {}
                ).items()
            },
            precision_support={
                str(precision).lower(): dict(support or {})
                for precision, support in (
                    data.get("precision_support") or {}
                ).items()
            },
            profile_revision=str(data.get("profile_revision", "")),
            audit_evidence=dict(data.get("audit_evidence") or {}),
            precision_aliases={
                str(k).lower(): str(v).lower()
                for k, v in (data.get("precision_aliases") or {}).items()
            },
            clock_hz=(
                float(data["clock_hz"]) if data.get("clock_hz") else None
            ),
            source=str(data.get("source") or source or "") or None,
            memory_hierarchy=tuple(
                MemoryLevel.load(item)
                for item in (data.get("memory_hierarchy") or [])
            ),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Validate all required architecture and audit invariants."""
        if not self.name:
            raise ValueError("architecture name is required")
        if self.vendor.upper() != "AMD":
            raise ValueError(
                "SOLAR-ROCm accepts AMD architecture profiles only",
            )
        if self.memory_bandwidth_bytes_per_second <= 0:
            raise ValueError("memory bandwidth must be positive")
        if not self.peak_ops_per_second or any(
            value <= 0 for value in self.peak_ops_per_second.values()
        ):
            raise ValueError(
                "at least one positive peak throughput is required",
            )
        from solar.schema_versions import AMD_RESOURCE_MODEL_VERSION

        if self.resource_model_version != AMD_RESOURCE_MODEL_VERSION:
            raise ValueError(
                "architecture resource_model_version must be "
                f"{AMD_RESOURCE_MODEL_VERSION}",
            )
        self._validate_resource_limits()
        self._validate_precision_support()
        if not self.profile_revision:
            raise ValueError("architecture profile_revision is required")
        _validate_audit_evidence_config(self.audit_evidence)
        self._validate_memory_hierarchy()

    def _validate_resource_limits(self) -> None:
        required = {
            "mfma",
            "valu",
            "sfu",
            "reduction",
            "atomic",
            "scan_sort",
            "conversion",
        }
        if set(self.resource_limits) != required:
            missing = sorted(required - set(self.resource_limits))
            extra = sorted(set(self.resource_limits) - required)
            raise ValueError(
                "resource_limits must define the complete AMD resource set; "
                f"missing={missing}, extra={extra}",
            )
        for resource_name, modes in self.resource_limits.items():
            if not modes or any(value <= 0 for value in modes.values()):
                raise ValueError(
                    f"resource limit rates for {resource_name} must be positive",
                )
            if resource_name not in self.resource_limit_sources:
                raise ValueError(
                    f"resource limit source is required for {resource_name}",
                )
        for (
            resource_name,
            exempt_modes,
        ) in self.calibration_exempt_modes.items():
            if resource_name not in self.resource_limits:
                raise ValueError(
                    f"calibration exemption has unknown resource {resource_name}",
                )
            for mode, reason in exempt_modes.items():
                if (
                    mode not in self.resource_limits[resource_name]
                    or not reason
                ):
                    raise ValueError(
                        "calibration exemptions require a declared mode and reason",
                    )

    def _validate_precision_support(self) -> None:
        if set(self.precision_support) != set(self.peak_ops_per_second):
            missing = sorted(
                set(self.peak_ops_per_second) - set(self.precision_support),
            )
            extra = sorted(
                set(self.precision_support) - set(self.peak_ops_per_second),
            )
            raise ValueError(
                "precision_support must describe every published peak "
                f"precision; missing={missing}, extra={extra}",
            )
        required_fields = {
            "hardware",
            "software",
            "calibration",
            "evidence",
        }
        for precision, support in self.precision_support.items():
            if not required_fields.issubset(support):
                raise ValueError(
                    f"precision_support.{precision} must define "
                    f"{sorted(required_fields)}",
                )
            if support["calibration"] not in {"required", "exempt"}:
                raise ValueError(
                    f"precision_support.{precision}.calibration must be "
                    "required or exempt",
                )
            if not all(
                str(support[field]).strip() for field in required_fields
            ):
                raise ValueError(
                    f"precision_support.{precision} fields must be non-empty",
                )
            if (
                support["calibration"] == "exempt"
                and not str(support.get("limitation", "")).strip()
            ):
                raise ValueError(
                    f"precision_support.{precision} exemption requires "
                    "a limitation",
                )

    def _validate_memory_hierarchy(self) -> None:
        names = [level.name for level in self.memory_hierarchy]
        if len(names) != len(set(names)):
            raise ValueError("memory hierarchy level names must be unique")
        for level in self.memory_hierarchy:
            if not level.name or not level.scope:
                raise ValueError(
                    "memory hierarchy levels require name and scope",
                )
            if level.capacity_bytes is not None and level.capacity_bytes <= 0:
                raise ValueError("memory hierarchy capacities must be positive")
            if (
                level.bandwidth_bytes_per_second is not None
                and level.bandwidth_bytes_per_second <= 0
            ):
                raise ValueError("memory hierarchy bandwidths must be positive")

    def require_verified_audit_evidence(self) -> None:
        """Reject formal use when the resource audit is not verified and present."""
        if self.audit_evidence.get("status") != "verified":
            reason = self.audit_evidence.get("reason_code", "not_verified")
            raise ValueError(
                f"architecture audit evidence unavailable: {reason}",
            )
        path_value = str(self.audit_evidence.get("path", ""))
        if not path_value:
            raise ValueError(
                "verified architecture audit evidence lacks a path",
            )
        profile_path = _packaged_profile_path(self.name)
        evidence_path = profile_path.parents[2] / path_value
        verify_resource_peak_audit(
            evidence_path,
            expected_sha256=str(self.audit_evidence["sha256"]),
            expected_schema_version=str(
                self.audit_evidence["required_schema_version"],
            ),
            expected_timing_profile=str(
                self.audit_evidence["required_timing_profile"],
            ),
            expected_clocks_locked=bool(
                self.audit_evidence["required_clocks_locked"],
            ),
            expected_unthrottled=bool(
                self.audit_evidence["required_unthrottled"],
            ),
            expected_gfx_target=str(self.audit_evidence["gfx_target"]),
            expected_precisions=self.required_calibration_precisions(),
            expected_resource_modes=self.required_calibration_resource_modes(),
            expected_instruction_checks=tuple(
                sorted(
                    str(item)
                    for item in self.audit_evidence[
                        "required_instruction_checks"
                    ]
                ),
            ),
        )

    def required_calibration_precisions(self) -> tuple[str, ...]:
        """Return every published precision that requires runtime calibration."""
        return tuple(
            sorted(
                precision
                for precision, support in self.precision_support.items()
                if support["calibration"] == "required"
            ),
        )

    def required_calibration_resource_modes(self) -> tuple[str, ...]:
        """Return every resource mode not covered by an explicit exemption."""
        return tuple(
            sorted(
                f"{resource}/{mode}"
                for resource, modes in self.resource_limits.items()
                for mode in modes
                if mode not in self.calibration_exempt_modes.get(resource, {})
            ),
        )

    def normalize_precision(self, precision: str) -> str:
        """Resolve spelling and vendor-specific format aliases."""
        key = _PRECISION_ALIASES.get(precision.lower(), precision.lower())
        return self.precision_aliases.get(key, key)

    def tensor_precision(
        self,
        dtype: object,
        fallback: str | None = None,
    ) -> str:
        """Resolve a tensor dtype without merging incompatible vendor formats."""
        key = str(dtype or "").strip().lower()
        key = key.removeprefix("torch.")
        if key in _VENDOR_SPECIFIC_DTYPES:
            precision = self.normalize_precision(key)
            if precision == key or precision not in self.peak_ops_per_second:
                raise ValueError(
                    f"Tensor dtype {key!r} is not supported by {self.name}",
                )
            return precision
        return normalize_dtype(key, fallback)

    def resource_rate_for(self, resource: str, mode: str) -> float:
        """Return the declared architectural upper rate for one resource mode."""
        resource_name = str(resource).lower()
        mode_name = str(mode).lower()
        try:
            modes = self.resource_limits[resource_name]
        except KeyError as exc:
            raise ValueError(
                f"Resource {resource!r} is not supported by {self.name}",
            ) from exc
        if mode_name in modes:
            return modes[mode_name]
        if "generic" in modes:
            return modes["generic"]
        raise ValueError(
            f"Resource mode {resource_name}:{mode_name} is not supported by {self.name}",
        )

    def resource_seconds(
        self,
        resource_work: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        """Reduce graph counters to per-pipeline times.

        Work sharing a resource is serialized and summed.  Independent AMD
        resources may overlap, so callers take the maximum across resources.
        """
        return {
            str(resource): sum(
                float(amount) / self.resource_rate_for(str(resource), str(mode))
                for mode, amount in modes.items()
            )
            for resource, modes in resource_work.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible architecture profile."""
        return {
            "name": self.name,
            "vendor": self.vendor,
            "gfx_target": self.gfx_target,
            "compute_units": self.compute_units,
            "memory_capacity_bytes": self.memory_capacity_bytes,
            "memory_bandwidth_bytes_per_second": self.memory_bandwidth_bytes_per_second,
            "l2_bytes": self.l2_bytes,
            "last_level_cache_bytes": self.last_level_cache_bytes,
            "peak_ops_per_second": dict(self.peak_ops_per_second),
            "resource_model_version": self.resource_model_version,
            "resource_limits": {
                resource: dict(modes)
                for resource, modes in self.resource_limits.items()
            },
            "resource_limit_sources": dict(self.resource_limit_sources),
            "calibration_exempt_modes": {
                resource: dict(modes)
                for resource, modes in self.calibration_exempt_modes.items()
            },
            "precision_support": {
                precision: dict(support)
                for precision, support in self.precision_support.items()
            },
            "profile_revision": self.profile_revision,
            "audit_evidence": dict(self.audit_evidence),
            "precision_aliases": dict(self.precision_aliases),
            "clock_hz": self.clock_hz,
            "source": self.source,
            "memory_hierarchy": [
                {
                    "name": level.name,
                    "scope": level.scope,
                    "capacity_bytes": level.capacity_bytes,
                    "bandwidth_bytes_per_second": level.bandwidth_bytes_per_second,
                    "source": level.source,
                }
                for level in self.memory_hierarchy
            ],
        }
