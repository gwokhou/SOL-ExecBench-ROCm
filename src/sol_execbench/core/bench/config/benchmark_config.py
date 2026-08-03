# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned configuration for benchmark execution."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.data.base_model import CurrentSchemaModel
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)


class BenchmarkConfig(CurrentSchemaModel):
    """Configuration for benchmark runs."""

    current_schema_version = SchemaVersion.BENCHMARK_CONFIG

    schema_version: Literal[SchemaVersion.BENCHMARK_CONFIG] = (
        SchemaVersion.BENCHMARK_CONFIG
    )
    warmup_runs: int = Field(default=10, ge=0)
    iterations: int = Field(default=50, gt=0)
    trials: int = Field(default=3, gt=0)
    min_measurement_time_seconds: float | None = Field(default=None, gt=0)
    lock_clocks: bool = True
    benchmark_reference: bool = True
    seed: int = 200

    @model_validator(mode="after")
    def _timing_settings_are_valid(self) -> BenchmarkConfig:
        """Keep explicit validation messages stable for CLI callers."""
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.trials <= 0:
            raise ValueError("trials must be > 0")
        if (
            self.min_measurement_time_seconds is not None
            and self.min_measurement_time_seconds <= 0
        ):
            raise ValueError("min_measurement_time_seconds must be > 0 or None")
        return self

    @property
    def timing_protocol(self) -> str:
        """Return the declared protocol, distinguishing custom diagnostic runs."""
        if (
            self.warmup_runs == 10
            and self.iterations == 50
            and self.trials == 3
            and self.min_measurement_time_seconds is None
            and self.lock_clocks
        ):
            return SchemaVersion.ROCM_EVENT_TIMING_PAPER_COUNTS
        return SchemaVersion.ROCM_EVENT_TIMING_CUSTOM
