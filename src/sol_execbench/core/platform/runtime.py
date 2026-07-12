# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment inspection helpers."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sol_execbench.core.data.trace import Environment


def env_snapshot(device: str) -> "Environment":
    """Collect the hardware and library information for *device*."""

    import torch

    from sol_execbench.core.data.trace import Environment

    libs: Dict[str, str] = {"torch": torch.__version__}
    try:
        import triton as _tr

        libs["triton"] = getattr(_tr, "__version__", "unknown")
    except Exception:
        pass

    try:
        import torch.version as tv

        if hip_version := getattr(tv, "hip", None):
            libs["hip"] = str(hip_version)
            libs["rocm"] = str(hip_version)
        elif cuda_version := getattr(tv, "cuda", None):
            libs["cuda"] = str(cuda_version)
    except Exception:
        pass
    return Environment(hardware=hardware_from_device(device), libs=libs)


def hardware_from_device(device: str) -> str:
    """Return a human-readable hardware name for a Torch device."""

    import torch

    parsed_device = torch.device(device)
    if parsed_device.type == "cuda":
        return torch.cuda.get_device_name(parsed_device.index)
    if parsed_device.type == "cpu":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or platform.machine() or "CPU"
    if parsed_device.type == "mps":
        return "Apple GPU (MPS)"
    if parsed_device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return torch.xpu.get_device_name(parsed_device.index)
        except Exception:
            return "Intel XPU"
    return parsed_device.type
