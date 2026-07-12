# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public AMD hardware-model API and external model reader.

Strict payload parsing and validation live in the private
``_amd_hardware_model_parsing`` module so this high-inbound public facade stays
focused on compatibility imports and resource loading.
"""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.scoring._amd_hardware_model_parsing import (
    AMD_HARDWARE_MODEL_SCHEMA_VERSION,
    AmdHardwareModel,
    EstimateConfidence,
    HardwareProfile,
    HardwareValidationStatus,
    amd_hardware_model_from_dict,
)

__all__ = [
    "AMD_HARDWARE_MODEL_SCHEMA_VERSION",
    "AmdHardwareModel",
    "EstimateConfidence",
    "HardwareProfile",
    "HardwareValidationStatus",
    "amd_hardware_model_from_dict",
    "load_amd_hardware_model",
]


def load_amd_hardware_model(path: Path) -> AmdHardwareModel:
    """Load and strictly validate a hardware model from an external JSON file."""
    return amd_hardware_model_from_dict(
        json.loads(path.read_text(encoding="utf-8")), source=str(path)
    )
