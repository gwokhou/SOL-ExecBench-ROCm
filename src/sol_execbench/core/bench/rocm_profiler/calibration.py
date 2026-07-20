# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict current schema for rocprofv3 overhead calibration artifacts."""

from typing import Any, Literal

from pydantic import Field

from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.integrity.schema_versions import (
    ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION,
)


class CalibrationClockSetup(StrictArtifactModel):
    """Clock-management state captured during calibration."""

    managed: bool
    lock_acquired: bool
    reset_on_exit: bool


class Rocprofv3OverheadCalibration(StrictArtifactModel):
    """One current profiler-overhead calibration artifact."""

    schema_version: Literal["sol_execbench.rocprofv3_overhead_calibration.v1"] = (
        ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION
    )
    generated_at: str
    baseline_median_ms: float
    profiler_median_ms: float
    overhead_ms: float
    iterations: int = Field(gt=0)
    warmup_runs: int = Field(ge=0)
    element_count: int = Field(gt=0)
    gpu_architecture: str
    profiler_executable: str
    clock_locked: bool
    clock_setup: CalibrationClockSetup
    gpu_isolation: dict[str, Any]
    baseline_sample_count: int = Field(gt=0)
    profiler_sample_count: int = Field(gt=0)
    source_revision: str | None = None


__all__ = [
    "ROCPROFV3_OVERHEAD_CALIBRATION_SCHEMA_VERSION",
    "CalibrationClockSetup",
    "Rocprofv3OverheadCalibration",
]
