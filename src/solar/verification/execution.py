# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Semantic interpreter and runtime identities for replay and artifact reuse."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from enum import StrEnum
from itertools import chain
from pathlib import Path

from solar.artifacts import sha256_bytes, stable_json_checksum
from solar.verification.errors import VerificationError

SEMANTIC_SOURCE_FOLDERS = ("verification", "ir", "graph")
SEMANTIC_SOURCE_FILES = ("storage.py", "precision.py", "schema_versions.py")
MAX_SEMANTIC_SOURCE_FILES = 512
MAX_SEMANTIC_SOURCE_BYTES = 16 * 1024 * 1024


class ExecutionBackend(StrEnum):
    """Actual tensor backends accepted in a semantic execution record."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


def semantic_runtime_identity() -> dict[str, str]:
    """Bind the interpreter source and Python/Torch versions without GPU queries."""
    import torch

    root = Path(__file__).resolve().parents[1]
    paths = chain(
        *(
            root.joinpath(folder).rglob("*.py")
            for folder in SEMANTIC_SOURCE_FOLDERS
        ),
        (root / name for name in SEMANTIC_SOURCE_FILES),
    )
    digests: dict[str, str] = {}
    remaining = MAX_SEMANTIC_SOURCE_BYTES
    for path in paths:
        if len(digests) >= MAX_SEMANTIC_SOURCE_FILES:
            raise VerificationError(
                "semantic implementation exceeds the file budget"
            )
        with path.open("rb") as stream:
            payload = stream.read(remaining + 1)
        if len(payload) > remaining:
            raise VerificationError(
                "semantic implementation exceeds the byte budget"
            )
        remaining -= len(payload)
        digests[path.relative_to(root).as_posix()] = sha256_bytes(payload)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "semantic_implementation_sha256": stable_json_checksum(digests),
    }


def require_current_semantic_runtime(recorded: Mapping[str, object]) -> None:
    """Reject missing or changed runtime fields before reusing a semantic proof."""
    for key, expected in semantic_runtime_identity().items():
        if recorded.get(key) != expected:
            raise VerificationError(
                f"verification semantic runtime differs: {key}"
            )


def execution_identity(device: str) -> dict[str, str]:
    """Identify the actual execution backend and its semantic runtime."""
    return {**semantic_runtime_identity(), **_device_identity(device)}


def _device_identity(device: str) -> dict[str, str]:
    import torch

    if not str(device).startswith("cuda"):
        return {
            "device_type": "cpu",
            "backend": ExecutionBackend.CPU.value,
            "device": str(device),
        }
    if not torch.cuda.is_available():
        raise VerificationError(
            f"requested CUDA/HIP device is unavailable: {device}"
        )
    selected = torch.device(device)
    index = (
        selected.index
        if selected.index is not None
        else torch.cuda.current_device()
    )
    if index < 0 or index >= torch.cuda.device_count():
        raise VerificationError(
            f"requested CUDA/HIP device index is invalid: {device}"
        )
    properties = torch.cuda.get_device_properties(index)
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        gfx_target = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        if not gfx_target.startswith("gfx"):
            raise VerificationError(
                "HIP runtime selected a device without an AMD gfx target"
            )
        return {
            "device_type": "cuda",
            "backend": ExecutionBackend.ROCM.value,
            "device": f"cuda:{index}",
            "hip_version": str(hip_version),
            "device_name": str(properties.name),
            "gfx_target": gfx_target,
        }
    return {
        "device_type": "cuda",
        "backend": ExecutionBackend.CUDA.value,
        "device": f"cuda:{index}",
        "device_name": str(properties.name),
    }
