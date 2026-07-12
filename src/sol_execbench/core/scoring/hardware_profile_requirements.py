# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The exact hardware profiles a bound suite is allowed to consume.

The old calibration command measured an architecture-wide convenience set.  That
is useful diagnostics, but it does not say which measurements are sufficient for
a particular score.  This small, serialisable contract makes that relationship
explicit and is intentionally independent of a particular calibration run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate


HARDWARE_PROFILE_REQUIREMENTS_SCHEMA_VERSION = (
    "sol_execbench.hardware_profile_requirements.v1"
)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class HardwareProfileRequirements:
    """Exact compute/memory profile keys required by a scoring suite."""

    architecture: str
    required_profile_keys: tuple[str, ...]
    scope: str
    payload_sha256: str | None = None
    schema_version: str = HARDWARE_PROFILE_REQUIREMENTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HARDWARE_PROFILE_REQUIREMENTS_SCHEMA_VERSION:
            raise ValueError("unsupported hardware profile requirements schema")
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise ValueError("architecture must be non-empty")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must be non-empty")
        if not self.required_profile_keys or any(
            not isinstance(key, str) or not key.strip()
            for key in self.required_profile_keys
        ):
            raise ValueError("required_profile_keys must contain non-empty strings")
        if tuple(sorted(set(self.required_profile_keys))) != self.required_profile_keys:
            raise ValueError("required_profile_keys must be sorted and unique")
        expected = _digest(self._payload_dict())
        if self.payload_sha256 is not None and self.payload_sha256 != expected:
            raise ValueError("hardware profile requirements checksum does not match")
        object.__setattr__(self, "payload_sha256", expected)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "required_profile_keys": list(self.required_profile_keys),
            "scope": self.scope,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "payload_sha256": self.payload_sha256}


def requirements_from_estimates(
    *,
    architecture: str,
    estimates: Iterable[OperatorWorkEstimate],
    scope: str,
) -> HardwareProfileRequirements:
    """Derive the exact measured profiles referenced by bound estimates."""
    keys: set[str] = set()
    for estimate in estimates:
        if estimate.flops > 0.0:
            if not all(
                (
                    estimate.compute_operation,
                    estimate.input_dtype,
                    estimate.output_dtype,
                    estimate.compute_path,
                )
            ):
                raise ValueError(
                    f"{estimate.node_id} has FLOPs but no exact compute profile"
                )
            keys.add(
                "compute."
                f"{estimate.compute_operation}.{estimate.input_dtype}."
                f"{estimate.output_dtype}.{estimate.compute_path}"
            )
        if estimate.total_bytes > 0.0:
            if not all(
                (
                    estimate.memory_access,
                    estimate.input_dtype,
                    estimate.output_dtype,
                    estimate.memory_path,
                )
            ):
                raise ValueError(
                    f"{estimate.node_id} has traffic but no exact memory profile"
                )
            keys.add(
                "memory."
                f"{estimate.memory_access}.{estimate.input_dtype}."
                f"{estimate.output_dtype}.{estimate.memory_path}"
            )
    return HardwareProfileRequirements(
        architecture=architecture,
        required_profile_keys=tuple(sorted(keys)),
        scope=scope,
    )


def hardware_profile_requirements_from_dict(
    payload: Mapping[str, Any],
) -> HardwareProfileRequirements:
    """Strictly parse a requirements artifact."""
    expected = {
        "schema_version",
        "architecture",
        "required_profile_keys",
        "scope",
        "payload_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("hardware profile requirements has invalid fields")
    raw_keys = payload["required_profile_keys"]
    if not isinstance(raw_keys, list):
        raise ValueError("required_profile_keys must be a list")
    return HardwareProfileRequirements(
        schema_version=str(payload["schema_version"]),
        architecture=str(payload["architecture"]),
        required_profile_keys=tuple(str(key) for key in raw_keys),
        scope=str(payload["scope"]),
        payload_sha256=str(payload["payload_sha256"]),
    )
