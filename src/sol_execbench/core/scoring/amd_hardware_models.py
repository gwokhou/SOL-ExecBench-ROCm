# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict AMD hardware-model readers and exact calibration profile resolution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.confidence import EstimateConfidence


AMD_HARDWARE_MODEL_SCHEMA_VERSION = "sol_execbench.amd_hardware_model.v2"
AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION = "sol_execbench.amd_hardware_model.v3"


class HardwareValidationStatus(str, Enum):
    VALIDATED = "validated"
    PROVISIONAL = "provisional"
    UNVALIDATED = "unvalidated"


@dataclass(frozen=True)
class HardwareProfile:
    """One exact calibrated rate, or an explicit absence of that evidence."""

    key: str
    state: str
    value: float | None
    confidence: EstimateConfidence
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.evidence_ref.strip():
            raise ValueError("hardware profile key and evidence_ref must be non-empty")
        if self.state not in {"measured", "unavailable", "unknown"}:
            raise ValueError(
                "hardware profile state must be measured, unavailable, or unknown"
            )
        if self.state == "measured":
            if (
                self.value is None
                or not math.isfinite(float(self.value))
                or float(self.value) <= 0.0
            ):
                raise ValueError("measured hardware profile value must be positive")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError(
                "unavailable or unknown hardware profiles cannot have a value"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "state": self.state,
            "value": self.value,
            "confidence": self.confidence.value,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class AmdHardwareModel:
    """AMD model with legacy scalar evidence and v3 exact profile collections."""

    architecture: str
    dtype_or_path: str
    peak_tflops: float
    memory_bandwidth_gbps: float
    clock_assumptions: tuple[str, ...]
    source: str
    confidence: EstimateConfidence
    hardware_validation_status: HardwareValidationStatus
    model_validation_status: HardwareValidationStatus
    evidence_refs: tuple[str, ...]
    schema_version: str = AMD_HARDWARE_MODEL_SCHEMA_VERSION
    compute_profiles: tuple[HardwareProfile, ...] = ()
    memory_profiles: tuple[HardwareProfile, ...] = ()

    def resolve_compute(
        self, operation: str, input_dtype: str, output_dtype: str, path: str
    ) -> HardwareProfile | None:
        key = f"compute.{operation}.{input_dtype}.{output_dtype}.{path}"
        return next(
            (profile for profile in self.compute_profiles if profile.key == key), None
        )

    def resolve_memory(
        self, access: str, input_dtype: str, output_dtype: str, path: str
    ) -> HardwareProfile | None:
        key = f"memory.{access}.{input_dtype}.{output_dtype}.{path}"
        return next(
            (profile for profile in self.memory_profiles if profile.key == key),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = self.confidence.value
        payload["hardware_validation_status"] = self.hardware_validation_status.value
        payload["model_validation_status"] = self.model_validation_status.value
        payload["clock_assumptions"] = list(self.clock_assumptions)
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["compute_profiles"] = [
            profile.to_dict() for profile in self.compute_profiles
        ]
        payload["memory_profiles"] = [
            profile.to_dict() for profile in self.memory_profiles
        ]
        if self.schema_version == AMD_HARDWARE_MODEL_SCHEMA_VERSION:
            payload.pop("compute_profiles")
            payload.pop("memory_profiles")
        else:
            payload.pop("dtype_or_path")
            payload.pop("peak_tflops")
            payload.pop("memory_bandwidth_gbps")
        return payload


def _non_empty(payload: dict[str, Any], key: str, source: str | None) -> str:
    value = str(payload[key]).strip()
    if not value:
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be non-empty"
        )
    return value


def _positive(payload: dict[str, Any], key: str, source: str | None) -> float:
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be numeric"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be a positive number"
        )
    return value


def _strings(payload: dict[str, Any], key: str, source: str | None) -> tuple[str, ...]:
    raw = payload[key]
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be a list of strings"
        )
    return tuple(raw)


def _confidence(value: object, source: str | None) -> EstimateConfidence:
    try:
        return EstimateConfidence(str(value))
    except ValueError as exc:
        raise ValueError(
            f"{source or 'hardware model'} has invalid confidence '{value}'"
        ) from exc


def _status(value: object, key: str, source: str | None) -> HardwareValidationStatus:
    try:
        return HardwareValidationStatus(str(value))
    except ValueError as exc:
        raise ValueError(
            f"{source or 'hardware model'} has invalid {key} '{value}'"
        ) from exc


def _profiles(
    payload: dict[str, Any], key: str, source: str | None
) -> tuple[HardwareProfile, ...]:
    raw = payload[key]
    if not isinstance(raw, list):
        raise ValueError(f"{source or 'hardware model'} field '{key}' must be a list")
    profiles: list[HardwareProfile] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "key",
            "state",
            "value",
            "confidence",
            "evidence_ref",
        }:
            raise ValueError(
                f"{source or 'hardware model'} {key} entries must have exact profile fields"
            )
        profiles.append(
            HardwareProfile(
                item["key"],
                item["state"],
                item["value"],
                _confidence(item["confidence"], source),
                item["evidence_ref"],
            )
        )
    if len({profile.key for profile in profiles}) != len(profiles):
        raise ValueError(f"{source or 'hardware model'} {key} contains duplicate keys")
    return tuple(profiles)


def amd_hardware_model_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
    expected_architecture: str | None = None,
) -> AmdHardwareModel:
    """Create a hardware model, retaining v2 scalars as deliberately inexact evidence."""
    if not isinstance(payload, dict):
        raise ValueError("hardware model payload must be a JSON object")
    schema_version = _non_empty(payload, "schema_version", source)
    common = {
        "schema_version",
        "architecture",
        "clock_assumptions",
        "source",
        "confidence",
        "hardware_validation_status",
        "model_validation_status",
        "evidence_refs",
    }
    v2 = common | {"dtype_or_path", "peak_tflops", "memory_bandwidth_gbps"}
    v3 = common | {"compute_profiles", "memory_profiles"}
    expected = (
        v2
        if schema_version == AMD_HARDWARE_MODEL_SCHEMA_VERSION
        else v3
        if schema_version == AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION
        else set()
    )
    if not expected or set(payload) != expected:
        unknown = sorted(set(payload) - expected)
        missing = sorted(expected - set(payload))
        details = [
            *(f"unknown field(s): {', '.join(unknown)}" for _ in [0] if unknown),
            *(f"missing required field: {key}" for key in missing),
        ]
        raise ValueError(
            f"{source or 'hardware model'} "
            + ("; ".join(details) or "has unsupported schema_version")
        )
    architecture = _non_empty(payload, "architecture", source)
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(
            f"{source or 'hardware model'} architecture '{architecture}' does not match expected '{expected_architecture}'"
        )
    if schema_version == AMD_HARDWARE_MODEL_SCHEMA_VERSION:
        dtype_or_path = _non_empty(payload, "dtype_or_path", source)
        peak_tflops = _positive(payload, "peak_tflops", source)
        memory_bandwidth_gbps = _positive(payload, "memory_bandwidth_gbps", source)
        compute_profiles = ()
        memory_profiles = ()
    else:
        dtype_or_path = "legacy scalar unavailable"
        peak_tflops = 0.0
        memory_bandwidth_gbps = 0.0
        compute_profiles = _profiles(payload, "compute_profiles", source)
        memory_profiles = _profiles(payload, "memory_profiles", source)
    return AmdHardwareModel(
        architecture,
        dtype_or_path,
        peak_tflops,
        memory_bandwidth_gbps,
        _strings(payload, "clock_assumptions", source),
        _non_empty(payload, "source", source),
        _confidence(payload["confidence"], source),
        _status(
            payload["hardware_validation_status"], "hardware_validation_status", source
        ),
        _status(payload["model_validation_status"], "model_validation_status", source),
        _strings(payload, "evidence_refs", source),
        schema_version,
        compute_profiles,
        memory_profiles,
    )


def load_amd_hardware_model(path: Path) -> AmdHardwareModel:
    return amd_hardware_model_from_dict(
        json.loads(path.read_text(encoding="utf-8")), source=str(path)
    )


def load_packaged_amd_hardware_model(architecture: str) -> AmdHardwareModel:
    path = resources.files("sol_execbench.data.amd_hardware_models").joinpath(
        f"{architecture}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged AMD hardware model not found for architecture '{architecture}'"
        )
    return amd_hardware_model_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        source=f"packaged: {architecture}",
        expected_architecture=architecture,
    )


def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    return {"gfx1200": load_packaged_amd_hardware_model("gfx1200")}
