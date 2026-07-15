# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment inspection helpers."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sol_execbench.core.data.trace import Environment


Which = Callable[[str], str | None]


def resolve_tool_path(tool: str, *, which: Which = shutil.which) -> Path | None:
    """Return the resolved absolute path for a tool available on ``PATH``."""
    located = which(tool)
    return Path(located).resolve() if located is not None else None


def discover_rocm_root(
    *,
    environ: Mapping[str, str] | None = None,
    which: Which = shutil.which,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> Path | None:
    """Discover the active ROCm root without assuming ``/opt/rocm``.

    An explicitly configured ``ROCM_PATH`` takes precedence.  Otherwise, infer
    the root from the resolved HIP compiler path, then retain ``/opt/rocm`` as
    a compatibility fallback for conventional installations.
    """
    environment = os.environ if environ is None else environ
    configured = environment.get("ROCM_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if is_dir(candidate):
            return candidate.resolve()

    hipcc = resolve_tool_path("hipcc", which=which)
    if hipcc is not None and hipcc.parent.name == "bin":
        candidate = hipcc.parent.parent
        if is_dir(candidate):
            return candidate

    conventional = Path("/opt/rocm")
    return conventional.resolve() if is_dir(conventional) else None


def resolve_rocm_tool(
    tool: str,
    *,
    environ: Mapping[str, str] | None = None,
    which: Which = shutil.which,
    is_file: Callable[[Path], bool] = Path.is_file,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> Path | None:
    """Find a ROCm tool from ``PATH`` or the discovered ROCm installation."""
    path = resolve_tool_path(tool, which=which)
    if path is not None:
        return path

    root = discover_rocm_root(environ=environ, which=which, is_dir=is_dir)
    if root is None:
        return None
    candidate = root / "bin" / tool
    return candidate if is_file(candidate) else None


def resolve_rocm_tool_command(
    tool: str,
    *,
    environ: Mapping[str, str] | None = None,
    which: Which = shutil.which,
    is_file: Callable[[Path], bool] = Path.is_file,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> str:
    """Return an invocation path without resolving a sudoers-visible symlink.

    Unlike :func:`resolve_rocm_tool`, this preserves the exact path returned by
    ``PATH`` because sudoers command matching distinguishes a symlink from its
    resolved target. If no installed file is found, the bare tool name is
    returned so callers receive the normal ``FileNotFoundError`` behavior.
    """
    located = which(tool)
    if located is not None:
        return located
    root = discover_rocm_root(environ=environ, which=which, is_dir=is_dir)
    if root is not None:
        candidate = root / "bin" / tool
        if is_file(candidate):
            return str(candidate)
    return tool


def rocm_search_roots(
    *,
    environ: Mapping[str, str] | None = None,
    which: Which = shutil.which,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> tuple[Path, ...]:
    """Return ordered roots suitable for ROCm header and library discovery."""
    candidates = [discover_rocm_root(environ=environ, which=which, is_dir=is_dir)]
    candidates.extend((Path("/opt/rocm"), Path("/usr"), Path("/usr/local")))
    roots: list[Path] = []
    for candidate in candidates:
        if candidate is None or not is_dir(candidate):
            continue
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


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
