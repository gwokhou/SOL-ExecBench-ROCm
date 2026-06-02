# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict AMD hardware model loader and v2 contract objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any


AMD_HARDWARE_MODEL_SCHEMA_VERSION = "sol_execbench.amd_hardware_model.v2"
VALIDATED_GFX1200_ONLY = "gfx1200"


class EstimateConfidence(str, Enum):
    """Confidence level for hardware model estimates."""

    SUPPORTED = "supported"
    INEXACT = "inexact"
    UNSUPPORTED = "unsupported"


class HardwareValidationStatus(str, Enum):
    """Validation state for architecture and model metadata."""

    VALIDATED = "validated"
    PROVISIONAL = "provisional"
    UNVALIDATED = "unvalidated"


@dataclass(frozen=True)
class AmdHardwareModel:
    """AMD hardware model contract object for v2 payloads."""

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize as JSON-safe payload."""
        payload = asdict(self)
        payload["confidence"] = self.confidence.value
        payload["hardware_validation_status"] = self.hardware_validation_status.value
        payload["model_validation_status"] = self.model_validation_status.value
        payload["clock_assumptions"] = list(self.clock_assumptions)
        payload["evidence_refs"] = list(self.evidence_refs)
        return payload


def _extract_payload(payload: dict[str, Any], *, source: str | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("hardware model payload must be a JSON object")
    return payload


def _require_keys(payload: dict[str, Any], *, source: str | None) -> None:
    allowed = {
        "schema_version",
        "architecture",
        "dtype_or_path",
        "peak_tflops",
        "memory_bandwidth_gbps",
        "clock_assumptions",
        "source",
        "confidence",
        "hardware_validation_status",
        "model_validation_status",
        "evidence_refs",
    }
    payload_keys = set(payload.keys())
    unknown = sorted(payload_keys - allowed)
    if unknown:
        raise ValueError(
            f"{source or 'hardware model'} contains unknown field(s): {', '.join(unknown)}"
        )
    for required in allowed:
        if required not in payload:
            raise ValueError(f"{source or 'hardware model'} missing required field: {required}")


def _parse_confidence(payload: dict[str, Any], *, source: str | None) -> EstimateConfidence:
    raw = str(payload["confidence"])
    try:
        return EstimateConfidence(raw)
    except ValueError as exc:
        valid_values = ", ".join([value.value for value in EstimateConfidence])
        raise ValueError(
            f"{source or 'hardware model'} has invalid confidence '{raw}', expected one of: {valid_values}"
        ) from exc


def _parse_status(
    payload: dict[str, Any], key: str, *, source: str | None
) -> HardwareValidationStatus:
    raw = str(payload[key])
    try:
        return HardwareValidationStatus(raw)
    except ValueError as exc:
        valid_values = ", ".join([value.value for value in HardwareValidationStatus])
        raise ValueError(
            f"{source or 'hardware model'} has invalid {key} '{raw}', expected one of: {valid_values}"
        ) from exc


def _parse_non_empty_string(
    payload: dict[str, Any], key: str, *, source: str | None
) -> str:
    raw = str(payload[key]).strip()
    if not raw:
        raise ValueError(f"{source or 'hardware model'} field '{key}' must be non-empty")
    return raw


def _parse_positive_float(
    payload: dict[str, Any], key: str, *, source: str | None
) -> float:
    value = payload[key]
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source or 'hardware model'} field '{key}' must be numeric") from exc
    if value_f <= 0.0:
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be a positive number"
        )
    return value_f


def _parse_str_sequence(
    payload: dict[str, Any], key: str, *, source: str | None
) -> tuple[str, ...]:
    raw = payload[key]
    if not isinstance(raw, list):
        raise ValueError(
            f"{source or 'hardware model'} field '{key}' must be a list of strings"
        )
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"{source or 'hardware model'} field '{key}'[{index}] must be a string"
            )
        values.append(item)
    return tuple(values)


def amd_hardware_model_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
    expected_architecture: str | None = None,
) -> AmdHardwareModel:
    """Create an AMD hardware model from parsed JSON payload."""
    payload = _extract_payload(payload, source=source)
    _require_keys(payload, source=source)

    schema_version = _parse_non_empty_string(payload, "schema_version", source=source)
    architecture = _parse_non_empty_string(payload, "architecture", source=source)
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(
            f"{source or 'hardware model'} architecture '{architecture}' does not match expected '{expected_architecture}'"
        )

    dtype_or_path = _parse_non_empty_string(payload, "dtype_or_path", source=source)
    peak_tflops = _parse_positive_float(payload, "peak_tflops", source=source)
    memory_bandwidth_gbps = _parse_positive_float(
        payload, "memory_bandwidth_gbps", source=source
    )
    clock_assumptions = _parse_str_sequence(
        payload, "clock_assumptions", source=source
    )
    source_field = _parse_non_empty_string(payload, "source", source=source)
    confidence = _parse_confidence(payload, source=source)
    hardware_validation_status = _parse_status(
        payload, "hardware_validation_status", source=source
    )
    model_validation_status = _parse_status(
        payload, "model_validation_status", source=source
    )
    evidence_refs = _parse_str_sequence(payload, "evidence_refs", source=source)
    if architecture != VALIDATED_GFX1200_ONLY:
        if (
            hardware_validation_status == HardwareValidationStatus.VALIDATED
            or model_validation_status == HardwareValidationStatus.VALIDATED
        ):
            raise ValueError(
                f"{source or 'hardware model'} only gfx1200 may use validated status fields in v1.9"
            )

    return AmdHardwareModel(
        architecture=architecture,
        dtype_or_path=dtype_or_path,
        peak_tflops=peak_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        clock_assumptions=clock_assumptions,
        source=source_field,
        confidence=confidence,
        hardware_validation_status=hardware_validation_status,
        model_validation_status=model_validation_status,
        evidence_refs=evidence_refs,
        schema_version=schema_version,
    )


def load_amd_hardware_model(path: Path) -> AmdHardwareModel:
    """Load an AMD hardware model from an external JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return amd_hardware_model_from_dict(payload, source=str(path))


def load_packaged_amd_hardware_model(architecture: str) -> AmdHardwareModel:
    """Load a packaged AMD hardware model resource by architecture token."""
    path = resources.files("sol_execbench.data.amd_hardware_models").joinpath(
        f"{architecture}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged AMD hardware model not found for architecture '{architecture}'"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return amd_hardware_model_from_dict(
        payload,
        source=f"packaged: {architecture}",
        expected_architecture=architecture,
    )


def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    """Return the built-in v1.9 AMD hardware model catalog."""
    return {"gfx1200": load_packaged_amd_hardware_model("gfx1200")}
