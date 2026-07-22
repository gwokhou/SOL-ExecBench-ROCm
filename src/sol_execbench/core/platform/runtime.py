# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment inspection helpers."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from sol_execbench.core.data.trace import Environment


Which = Callable[[str], str | None]

FALLBACK_CACHE_CLEAR_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class RocmDeviceInfo:
    """Runtime properties for one visible PyTorch ROCm device."""

    device: str
    index: int
    name: str
    gfx_target: str
    total_memory_bytes: int
    l2_cache_bytes: int | None
    torch_version: str
    hip_version: str


@dataclass(frozen=True)
class CacheClearPolicy:
    """Resolved L2 eviction-buffer policy for one benchmark device."""

    detected_l2_bytes: int | None
    clear_buffer_bytes: int
    source: str
    fallback_reason: str | None = None


def derive_cache_clear_policy(l2_cache_bytes: int | None) -> CacheClearPolicy:
    """Use twice the detected L2, falling back to the historical 256 MiB."""

    if l2_cache_bytes is not None and l2_cache_bytes > 0:
        return CacheClearPolicy(
            detected_l2_bytes=l2_cache_bytes,
            clear_buffer_bytes=2 * l2_cache_bytes,
            source="torch_device_properties",
        )
    return CacheClearPolicy(
        detected_l2_bytes=None,
        clear_buffer_bytes=FALLBACK_CACHE_CLEAR_BYTES,
        source="fallback_default",
        fallback_reason="l2_cache_size_unavailable",
    )


def detect_rocm_device(
    device: str = "cuda:0", *, torch_module: Any | None = None
) -> RocmDeviceInfo:
    """Detect one concrete PyTorch ROCm device and its execution capacities."""

    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415

    parsed = torch_module.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"ROCm target device must use the cuda namespace: {device}")
    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    if hip_version is None or not torch_module.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    index = (
        parsed.index if parsed.index is not None else torch_module.cuda.current_device()
    )
    if index < 0 or index >= torch_module.cuda.device_count():
        raise ValueError(f"ROCm device index is out of range: {index}")
    properties = torch_module.cuda.get_device_properties(index)
    raw_gfx = getattr(properties, "gcnArchName", "") or getattr(
        properties, "gfx_arch_name", ""
    )
    gfx_target = str(raw_gfx).split(":", maxsplit=1)[0].strip().lower()
    if not gfx_target.startswith("gfx"):
        raise RuntimeError(
            f"ROCm device did not expose a concrete gfx target: {raw_gfx!r}"
        )
    raw_l2 = getattr(properties, "L2_cache_size", None)
    l2_cache_bytes = int(raw_l2) if raw_l2 is not None and int(raw_l2) > 0 else None
    return RocmDeviceInfo(
        device=f"cuda:{index}",
        index=index,
        name=str(properties.name),
        gfx_target=gfx_target,
        total_memory_bytes=int(properties.total_memory),
        l2_cache_bytes=l2_cache_bytes,
        torch_version=str(getattr(torch_module, "__version__", "")),
        hip_version=str(hip_version),
    )


def cache_clear_policy_for_device(device: str) -> CacheClearPolicy:
    """Detect the device L2 and resolve its benchmark cache-clear policy."""

    return derive_cache_clear_policy(detect_rocm_device(device).l2_cache_bytes)


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
    if is_file(candidate):
        return candidate
    llvm_candidate = root / "lib" / "llvm" / "bin" / tool
    return llvm_candidate if is_file(llvm_candidate) else None


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


def env_snapshot(
    device: str,
    *,
    clocks_locked: bool | None = None,
    timing_protocol: str | None = None,
) -> "Environment":
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
    isolation = "unknown"
    if os.environ.get("SOL_EXECBENCH_SANDBOXED") == "1":
        isolation = "container"
    elif os.environ.get("SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION") == "1":
        isolation = "unsafe_local"
    return Environment(
        hardware=hardware_from_device(device),
        libs=libs,
        execution_isolation=isolation,
        clocks_locked=clocks_locked,
        timing_protocol=timing_protocol,
    )


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
