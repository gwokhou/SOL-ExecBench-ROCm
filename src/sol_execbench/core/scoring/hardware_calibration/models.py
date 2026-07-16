# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON-safe hardware-calibration evidence models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from math import isfinite
import re
from typing import Any, Mapping, TypeGuard, cast

from sol_execbench.core.scoring.hardware_calibration.statistics import (
    select_conservative_value,
)

CALIBRATION_SCHEMA_VERSION = "sol_execbench.hardware_calibration.v3"
CALIBRATION_CANDIDATE_STATES = frozenset({"measured", "unavailable", "unknown"})
ISA_VALIDATION_STATES = frozenset(
    {"verified", "unavailable", "failed", "not_applicable"}
)


def _is_json_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_json_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_json_number(value: object, field_name: str) -> float:
    if not _is_json_number(value):
        raise ValueError(f"{field_name} must be a JSON number")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")
    return numeric_value


@dataclass(frozen=True)
class CalibrationIsaValidation:
    """ISA proof bound to one compiled calibration candidate."""

    status: str
    architecture: str
    expected_instruction: str | None = None
    expected_subgroup: str | None = None
    matched_instruction_count: int = 0
    code_object_path: str | None = None
    code_object_sha256: str | None = None
    disassembly_path: str | None = None
    disassembly_sha256: str | None = None
    spec_provenance: Mapping[str, str] = field(default_factory=dict)
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ISA_VALIDATION_STATES:
            raise ValueError("invalid ISA validation status")
        if not self.architecture:
            raise ValueError("ISA validation architecture must be non-empty")
        if self.matched_instruction_count < 0:
            raise ValueError("ISA matched instruction count must be non-negative")
        if self.status == "verified":
            if re.fullmatch(r"gfx[0-9a-z]+", self.architecture) is None:
                raise ValueError("verified ISA validation requires a gfx architecture")
            if not self.expected_instruction or self.matched_instruction_count <= 0:
                raise ValueError(
                    "verified ISA validation requires a matched instruction"
                )
            if not self.code_object_sha256 or not self.disassembly_sha256:
                raise ValueError("verified ISA validation requires artifact checksums")
        elif self.status in {"unavailable", "failed"} and not self.reason_code:
            raise ValueError("failed ISA validation requires a reason code")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "architecture": self.architecture,
            "expected_instruction": self.expected_instruction,
            "expected_subgroup": self.expected_subgroup,
            "matched_instruction_count": self.matched_instruction_count,
            "code_object_path": self.code_object_path,
            "code_object_sha256": self.code_object_sha256,
            "disassembly_path": self.disassembly_path,
            "disassembly_sha256": self.disassembly_sha256,
            "spec_provenance": dict(self.spec_provenance),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    """One calibration capability, including its explicit evidence state."""

    key: str
    state: str
    value: float | None
    unit: str | None
    samples: tuple[float, ...] = ()
    reason_code: str | None = None
    retained_samples: tuple[float, ...] = ()
    retained_spread: float | None = None
    isa_validation: CalibrationIsaValidation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise ValueError("candidate key must be a string")
        if not isinstance(self.state, str):
            raise ValueError("candidate state must be a string")
        if not self.key.strip():
            raise ValueError("candidate key must be non-empty")
        if self.state not in CALIBRATION_CANDIDATE_STATES:
            raise ValueError(
                "candidate state must be measured, unavailable, or unknown"
            )
        if self.state != "measured":
            if self.value is not None:
                raise ValueError("only measured candidates may have a value")
            if self.unit is not None:
                raise ValueError("only measured candidates may have a unit")
            if (
                self.samples
                or self.retained_samples
                or self.retained_spread is not None
            ):
                raise ValueError("only measured candidates may have samples")
            if not self.reason_code:
                raise ValueError(
                    "unavailable and unknown candidates require a reason_code"
                )
            return
        if self.value is None or not _is_json_number(self.value):
            raise ValueError("measured candidate value must be a JSON number")
        value = _require_json_number(self.value, "value")
        if value <= 0.0:
            raise ValueError("measured candidate value must be finite and positive")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("measured candidate unit must be non-empty")
        if not isinstance(self.samples, tuple):
            raise ValueError("samples must be a tuple of JSON numbers")
        normalized_samples = tuple(
            _require_json_number(sample, f"samples[{index}]")
            for index, sample in enumerate(self.samples)
        )
        selection = select_conservative_value(normalized_samples)
        if abs(value - selection.value) > 1e-12:
            raise ValueError(
                "measured candidate value must equal conservative sample value"
            )
        if (
            self.retained_samples
            and self.retained_samples != selection.retained_samples
        ):
            raise ValueError("retained samples do not match conservative selection")
        if (
            self.retained_spread is not None
            and abs(self.retained_spread - selection.spread) > 1e-12
        ):
            raise ValueError("retained spread does not match conservative selection")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "samples", normalized_samples)
        object.__setattr__(self, "retained_samples", selection.retained_samples)
        object.__setattr__(self, "retained_spread", selection.spread)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "state": self.state,
            "value": self.value,
            "unit": self.unit,
            "samples": list(self.samples),
            "reason_code": self.reason_code,
            "retained_samples": list(self.retained_samples),
            "retained_spread": self.retained_spread,
            "isa_validation": (
                self.isa_validation.to_dict()
                if self.isa_validation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class HardwareCalibrationArtifact:
    """Versioned calibration evidence artifact, never a default hardware model."""

    generated_at: str
    candidates: tuple[CalibrationCandidate, ...]
    collection_status: str
    validation_status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload_sha256: str | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")
        if not isinstance(self.generated_at, str):
            raise ValueError("generated_at must be a string")
        if not isinstance(self.collection_status, str):
            raise ValueError("collection_status must be a string")
        if not isinstance(self.validation_status, str):
            raise ValueError("validation_status must be a string")
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema: {self.schema_version}")
        try:
            datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
        if not self.collection_status:
            raise ValueError("collection_status must be non-empty")
        if not self.validation_status:
            raise ValueError("validation_status must be non-empty")
        keys = [candidate.key for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration candidate keys must be unique")
        requirements = self.metadata.get("profile_requirements")
        protocol = self.metadata.get("probe_protocol")
        if not isinstance(requirements, Mapping) or not isinstance(protocol, Mapping):
            raise ValueError(
                "calibration requires profile_requirements and probe_protocol"
            )
        required_keys = requirements.get("required_profile_keys")
        if not isinstance(required_keys, list) or not required_keys:
            raise ValueError("calibration profile_requirements require profile keys")
        if not isinstance(protocol.get("warmup_iterations"), int) or not isinstance(
            protocol.get("timed_samples"), int
        ):
            raise ValueError("calibration probe protocol is incomplete")
        # The checksum binds every byte of the semantic payload.  It deliberately
        # excludes itself so callers can verify a parsed artifact identically.
        try:
            expected_checksum = _artifact_payload_sha256(self._payload_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact metadata must be JSON serializable") from exc
        if self.payload_sha256 is not None:
            if not isinstance(self.payload_sha256, str) or (
                self.payload_sha256 != expected_checksum
            ):
                raise ValueError("calibration payload checksum does not match payload")
        object.__setattr__(self, "payload_sha256", expected_checksum)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "collection_status": self.collection_status,
            "validation_status": self.validation_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "payload_sha256": self.payload_sha256}


def _artifact_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    extras = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if extras or missing:
        details = [
            *(f"unknown field {key}" for key in extras),
            *(f"missing field {key}" for key in missing),
        ]
        raise ValueError(f"{label} has " + ", ".join(details))


def calibration_candidate_from_dict(payload: Mapping[str, Any]) -> CalibrationCandidate:
    if not isinstance(payload, Mapping):
        raise ValueError("calibration candidate must be an object")
    expected = {
        "key",
        "state",
        "value",
        "unit",
        "samples",
        "reason_code",
        "retained_samples",
        "retained_spread",
        "isa_validation",
    }
    _require_exact_keys(payload, expected, "calibration candidate")
    if not isinstance(payload["samples"], list) or not isinstance(
        payload["retained_samples"], list
    ):
        raise ValueError("candidate samples must be lists")
    key = _require_json_string(payload["key"], "key")
    state = _require_json_string(payload["state"], "state")
    value = (
        None
        if payload["value"] is None
        else _require_json_number(payload["value"], "value")
    )
    unit = payload["unit"]
    if unit is not None:
        unit = _require_json_string(unit, "unit")
    reason_code = payload["reason_code"]
    if reason_code is not None:
        reason_code = _require_json_string(reason_code, "reason_code")
    samples = tuple(
        _require_json_number(sample, f"samples[{index}]")
        for index, sample in enumerate(payload["samples"])
    )
    retained_samples = tuple(
        _require_json_number(sample, f"retained_samples[{index}]")
        for index, sample in enumerate(payload["retained_samples"])
    )
    retained_spread = payload["retained_spread"]
    if retained_spread is not None:
        retained_spread = _require_json_number(retained_spread, "retained_spread")
    isa_validation = _isa_validation_from_dict(payload["isa_validation"])
    return CalibrationCandidate(
        key,
        state,
        value,
        unit,
        samples,
        reason_code,
        retained_samples,
        retained_spread,
        isa_validation,
    )


def _isa_validation_from_dict(value: object) -> CalibrationIsaValidation | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("candidate isa_validation must be an object or null")
    payload = cast(Mapping[str, object], value)
    expected = {
        "status",
        "architecture",
        "expected_instruction",
        "expected_subgroup",
        "matched_instruction_count",
        "code_object_path",
        "code_object_sha256",
        "disassembly_path",
        "disassembly_sha256",
        "spec_provenance",
        "reason_code",
    }
    _require_exact_keys(payload, expected, "candidate isa_validation")
    provenance = payload["spec_provenance"]
    if not isinstance(provenance, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in provenance.items()
    ):
        raise ValueError("ISA spec provenance must contain string pairs")
    return CalibrationIsaValidation(
        status=_require_json_string(payload["status"], "isa_validation.status"),
        architecture=_require_json_string(
            payload["architecture"], "isa_validation.architecture"
        ),
        expected_instruction=_optional_string(payload["expected_instruction"]),
        expected_subgroup=_optional_string(payload["expected_subgroup"]),
        matched_instruction_count=_require_non_negative_int(
            payload["matched_instruction_count"], "matched_instruction_count"
        ),
        code_object_path=_optional_string(payload["code_object_path"]),
        code_object_sha256=_optional_string(payload["code_object_sha256"]),
        disassembly_path=_optional_string(payload["disassembly_path"]),
        disassembly_sha256=_optional_string(payload["disassembly_sha256"]),
        spec_provenance=dict(cast(Mapping[str, str], provenance)),
        reason_code=_optional_string(payload["reason_code"]),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _require_json_string(value, "optional string")


def _require_non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def hardware_calibration_artifact_from_dict(
    payload: Mapping[str, Any],
) -> HardwareCalibrationArtifact:
    if not isinstance(payload, Mapping):
        raise ValueError("hardware calibration artifact must be an object")
    expected = {
        "schema_version",
        "generated_at",
        "metadata",
        "candidates",
        "collection_status",
        "validation_status",
        "payload_sha256",
    }
    # Pre-checksum diagnostics remain parseable; the model-build authority
    # boundary separately requires the checksum field.
    extras = sorted(set(payload) - expected)
    missing = sorted((expected - {"payload_sha256"}) - set(payload))
    if extras or missing:
        details = [
            *(f"unknown field {key}" for key in extras),
            *(f"missing field {key}" for key in missing),
        ]
        raise ValueError("hardware calibration artifact has " + ", ".join(details))
    if not isinstance(payload["metadata"], dict) or not isinstance(
        payload["candidates"], list
    ):
        raise ValueError(
            "artifact metadata must be an object and candidates must be a list"
        )
    return HardwareCalibrationArtifact(
        schema_version=_require_json_string(
            payload["schema_version"], "schema_version"
        ),
        generated_at=_require_json_string(payload["generated_at"], "generated_at"),
        metadata=payload["metadata"],
        candidates=tuple(
            calibration_candidate_from_dict(item) for item in payload["candidates"]
        ),
        collection_status=_require_json_string(
            payload["collection_status"], "collection_status"
        ),
        validation_status=_require_json_string(
            payload["validation_status"], "validation_status"
        ),
        payload_sha256=(
            _require_json_string(payload["payload_sha256"], "payload_sha256")
            if "payload_sha256" in payload
            else None
        ),
    )
