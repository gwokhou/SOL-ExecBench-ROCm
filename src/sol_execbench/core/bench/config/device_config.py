# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Device-specific configuration for SOL ExecBench benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClockPreset:
    """ROCm DPM levels for stable benchmarking."""

    sclk_level: int
    mclk_level: int


CLOCK_LOCK_PRESETS: dict[str, ClockPreset] = {
    "gfx1200": ClockPreset(sclk_level=1, mclk_level=1),
    "gfx942": ClockPreset(sclk_level=1, mclk_level=1),
    "AMD Radeon": ClockPreset(sclk_level=1, mclk_level=1),
    "AMD Instinct": ClockPreset(sclk_level=1, mclk_level=1),
}


def get_clock_preset(device_name: str) -> Optional[ClockPreset]:
    """Get the ROCm clock preset for a GPU device name or architecture string."""
    for key, preset in CLOCK_LOCK_PRESETS.items():
        if key.lower() in device_name.lower():
            return preset
    return None
