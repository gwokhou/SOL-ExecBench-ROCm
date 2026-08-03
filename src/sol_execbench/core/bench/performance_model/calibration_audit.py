# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict wire contract for diagnostic calibration audit evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CalibrationMetric(StrictArtifactModel):
    """One positive metric emitted by a calibration probe."""

    model_config = _CONFIG

    name: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    value: float = Field(gt=0)
    unit: str = Field(min_length=1)


class CalibrationProbeBatch(StrictArtifactModel):
    """One fresh-process batch in the two-phase calibration protocol."""

    model_config = _CONFIG

    phase: str = Field(min_length=1)
    process_batch: int = Field(ge=0)
    mode: str = Field(min_length=1)
    clocks_locked: bool
    metrics: list[CalibrationMetric] = Field(min_length=1)


class CalibrationProbeIdentity(StrictArtifactModel):
    """Content and platform identity of the compiled calibration probe."""

    model_config = _CONFIG

    source_sha256: SHA256Digest
    binary_sha256: SHA256Digest
    compiler_sha256: SHA256Digest
    architecture: str = Field(min_length=1)
    rocm_version: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    gpu_id: str = Field(min_length=1)
    gpu_bdf: str = Field(min_length=1)
    total_memory_bytes: int = Field(gt=0)
    compiler_version: str = Field(min_length=1)
    isa: dict[str, Any]


class CalibrationProtocol(StrictArtifactModel):
    """Frozen two-phase statistical protocol used by the audit."""

    model_config = _CONFIG

    design: Literal["two_phase_tuning_then_parameter_estimation"]
    configuration_frozen_before_parameter_estimation: Literal[True]
    tuning_process_batches: int = Field(gt=0)
    parameter_estimation_process_batches: int = Field(ge=5)
    bootstrap_replicates: int = Field(gt=0)
    bootstrap_seed: int = Field(ge=0)
    clock_mode: Literal["STABLE_PEAK"]


class DiagnosticCalibrationAudit(CurrentSchemaModel):
    """Current diagnostic calibration audit artifact."""

    model_config = _CONFIG
    current_schema_version = SchemaVersion.DIAGNOSTIC_CALIBRATION_AUDIT

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_CALIBRATION_AUDIT] = (
        SchemaVersion.DIAGNOSTIC_CALIBRATION_AUDIT
    )
    probe_identity: CalibrationProbeIdentity
    protocol: CalibrationProtocol
    frozen_configuration: dict[str, str]
    tuning_evidence: list[CalibrationProbeBatch] = Field(min_length=1)
    parameter_estimation_evidence: list[CalibrationProbeBatch] = Field(
        min_length=1,
    )
    environment: list[RuntimeGPUTelemetry] = Field(min_length=2)

    @model_validator(mode="after")
    def environment_is_stable(self) -> DiagnosticCalibrationAudit:
        """Require stable pre/post identity without foreign GPU work."""
        if {item.phase for item in self.environment} != {"pre", "post"}:
            raise ValueError("calibration requires pre/post environment")
        identity = self.probe_identity
        for item in self.environment:
            if (
                item.gpu_id != identity.gpu_id
                or item.gpu_bdf != identity.gpu_bdf
            ):
                raise ValueError("calibration environment identity mismatch")
            if item.performance_level != "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK":
                raise ValueError("calibration clock mode is not stable peak")
            if item.temperature_c is None:
                raise ValueError("calibration temperature is unavailable")
            if item.foreign_process_count not in {0, None}:
                raise ValueError("calibration foreign GPU process detected")
        return self


__all__ = ["DiagnosticCalibrationAudit"]
