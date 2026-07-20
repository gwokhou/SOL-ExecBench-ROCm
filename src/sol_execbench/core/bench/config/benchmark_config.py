# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS


OFFICIAL_ROCM_TIMING_PROTOCOL = SCHEMA_VERSIONS["rocm_event_timing_paper_counts"]
CUSTOM_ROCM_TIMING_PROTOCOL = SCHEMA_VERSIONS["rocm_event_timing_custom"]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    All fields have default values to make configuration optional.
    """

    warmup_runs: int = field(default=10)
    iterations: int = field(default=50)
    trials: int = field(default=3)
    min_measurement_time_seconds: float | None = field(default=None)
    lock_clocks: bool = field(default=True)
    benchmark_reference: bool = field(default=True)
    seed: int = field(default=200)

    def __post_init__(self):
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
            return OFFICIAL_ROCM_TIMING_PROTOCOL
        return CUSTOM_ROCM_TIMING_PROTOCOL
