# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public AMD hardware-model API and packaged model readers.

Strict payload parsing and validation live in the private
``_amd_hardware_model_parsing`` module so this high-inbound public facade stays
focused on compatibility imports and resource loading.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from sol_execbench.core.scoring._amd_hardware_model_parsing import (
    AMD_HARDWARE_MODEL_SCHEMA_VERSION,
    AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION,
    AmdHardwareModel,
    EstimateConfidence,
    HardwareProfile,
    HardwareValidationStatus,
    amd_hardware_model_from_dict,
)

__all__ = [
    "AMD_HARDWARE_MODEL_SCHEMA_VERSION",
    "AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION",
    "AmdHardwareModel",
    "EstimateConfidence",
    "HardwareProfile",
    "HardwareValidationStatus",
    "amd_hardware_model_from_dict",
    "default_amd_hardware_models",
    "load_amd_hardware_model",
    "load_packaged_amd_hardware_model",
]


def load_amd_hardware_model(path: Path) -> AmdHardwareModel:
    """Load and strictly validate a hardware model from an external JSON file."""
    return amd_hardware_model_from_dict(
        json.loads(path.read_text(encoding="utf-8")), source=str(path)
    )


def load_packaged_amd_hardware_model(architecture: str) -> AmdHardwareModel:
    """Load the packaged model for one exact AMD architecture."""
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
    """Return the repository's packaged default hardware-model set."""
    return {"gfx1200": load_packaged_amd_hardware_model("gfx1200")}
