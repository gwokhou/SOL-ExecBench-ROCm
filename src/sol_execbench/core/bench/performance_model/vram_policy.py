# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Frozen total-VRAM policy for diagnostic working-set calibration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion

MIB = 1 << 20
GIB = 1 << 30
CAPACITY_TOLERANCE_FRACTION = 0.05
VRAM_APPLICABILITY_MIN_BYTES = 64 * MIB
VRAM_POLICY_ALGORITHM = "rdna4_total_memory_class.v1"

_CAPACITY_CLASSES = {
    8 * GIB: 256 * MIB,
    16 * GIB: 512 * MIB,
}
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
_NominalCapacity = Literal[8589934592, 17179869184]
_ProbeWorkingSet = Literal[268435456, 536870912]


class DiagnosticVRAMWorkingSetPolicy(CurrentSchemaModel):
    """Immutable capacity class selected before successor design."""

    model_config = _CONFIG
    current_schema_version = SchemaVersion.DIAGNOSTIC_VRAM_WORKING_SET_POLICY

    schema_version: Literal[
        SchemaVersion.DIAGNOSTIC_VRAM_WORKING_SET_POLICY
    ] = SchemaVersion.DIAGNOSTIC_VRAM_WORKING_SET_POLICY
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    gpu_architecture: Literal["gfx1200"]
    gpu_id: NonEmptyString
    observed_total_memory_bytes: int = Field(gt=0)
    nominal_total_memory_bytes: _NominalCapacity
    probe_working_set_bytes: _ProbeWorkingSet
    applicability_min_bytes: Literal[67108864] = VRAM_APPLICABILITY_MIN_BYTES
    applicability_max_bytes: _ProbeWorkingSet
    selection_algorithm: Literal["rdna4_total_memory_class.v1"] = (
        VRAM_POLICY_ALGORITHM
    )
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    frozen_before_design: Literal[True] = True
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _selection_is_canonical(self) -> DiagnosticVRAMWorkingSetPolicy:
        expected_probe = _CAPACITY_CLASSES[self.nominal_total_memory_bytes]
        if (
            self.probe_working_set_bytes != expected_probe
            or self.applicability_max_bytes != expected_probe
        ):
            raise ValueError(
                "VRAM policy probe tier differs from capacity class"
            )
        deviation = abs(
            self.observed_total_memory_bytes - self.nominal_total_memory_bytes
        )
        if deviation > (
            self.nominal_total_memory_bytes * CAPACITY_TOLERANCE_FRACTION
        ):
            raise ValueError("VRAM policy observed capacity differs from class")
        return self


def select_vram_working_set_policy(
    *,
    gpu_architecture: str,
    gpu_id: str,
    total_memory_bytes: int,
    source_revision: str,
    created_at: str | None = None,
) -> DiagnosticVRAMWorkingSetPolicy:
    """Select the only supported RDNA4 capacity tier, or fail closed."""
    if gpu_architecture != "gfx1200":
        raise ValueError(
            f"unsupported_vram_policy_architecture:{gpu_architecture}"
        )
    candidates = [
        nominal
        for nominal in _CAPACITY_CLASSES
        if abs(total_memory_bytes - nominal)
        <= nominal * CAPACITY_TOLERANCE_FRACTION
    ]
    if len(candidates) != 1:
        raise ValueError(f"unsupported_total_vram_class:{total_memory_bytes}")
    nominal = cast("_NominalCapacity", candidates[0])
    probe = cast("_ProbeWorkingSet", _CAPACITY_CLASSES[nominal])
    return DiagnosticVRAMWorkingSetPolicy(
        gpu_architecture="gfx1200",
        gpu_id=gpu_id,
        observed_total_memory_bytes=total_memory_bytes,
        nominal_total_memory_bytes=nominal,
        probe_working_set_bytes=probe,
        applicability_max_bytes=probe,
        source_revision=source_revision,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )


__all__ = [
    "CAPACITY_TOLERANCE_FRACTION",
    "GIB",
    "MIB",
    "VRAM_APPLICABILITY_MIN_BYTES",
    "VRAM_POLICY_ALGORITHM",
    "DiagnosticVRAMWorkingSetPolicy",
    "select_vram_working_set_policy",
]
