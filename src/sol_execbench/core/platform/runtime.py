# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment inspection helpers."""

from __future__ import annotations

import os
import platform
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_SANDBOXED,
    ENV_SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION,
)

if TYPE_CHECKING:
    from sol_execbench.core.data.trace import Environment


Which = Callable[[str], str | None]

FALLBACK_CACHE_CLEAR_BYTES = 256 * 1024 * 1024
PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
SYS_DEVICES_ROOT = Path("/sys/devices")
_PCI_BDF_PATTERN = r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
_PCI_BDF_RE = re.compile(f"^{_PCI_BDF_PATTERN}$")
_PCIE_SPEED_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s+GT/s(?:\s+PCIe)?$")
_PCIE_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
)


class PCIeLinkIdentity(StrictArtifactModel):
    """Negotiated and maximum identity of one PCIe link endpoint."""

    model_config = _PCIE_CONFIG

    bdf: str = Field(pattern=f"^{_PCI_BDF_PATTERN}$")
    current_speed_gtps: float = Field(gt=0)
    max_speed_gtps: float = Field(gt=0)
    current_width: int = Field(gt=0)
    max_width: int = Field(gt=0)

    @model_validator(mode="after")
    def _negotiated_values_fit_capability(self) -> PCIeLinkIdentity:
        if self.current_speed_gtps > self.max_speed_gtps:
            raise ValueError("PCIe current speed exceeds maximum speed")
        if self.current_width > self.max_width:
            raise ValueError("PCIe current width exceeds maximum width")
        return self


class PCIeTopologyIdentity(StrictArtifactModel):
    """Ordered CPU-root-to-GPU PCIe path and its effective bottleneck."""

    model_config = _PCIE_CONFIG

    links: tuple[PCIeLinkIdentity, ...] = Field(min_length=1)
    bottleneck_bdf: str = Field(pattern=f"^{_PCI_BDF_PATTERN}$")
    effective_speed_gtps: float = Field(gt=0)
    effective_width: int = Field(gt=0)

    @property
    def endpoint_bdf(self) -> str:
        """Return the final device BDF in the ordered path."""
        return self.links[-1].bdf

    @field_validator("links", mode="before")
    @classmethod
    def _json_links_are_immutable(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _effective_link_is_derived(self) -> PCIeTopologyIdentity:
        bdfs = [link.bdf for link in self.links]
        if len(bdfs) != len(set(bdfs)):
            raise ValueError("PCIe topology repeats a BDF")
        bottleneck = min(
            self.links,
            key=lambda link: link.current_speed_gtps * link.current_width,
        )
        if (
            self.bottleneck_bdf != bottleneck.bdf
            or self.effective_speed_gtps != bottleneck.current_speed_gtps
            or self.effective_width != bottleneck.current_width
        ):
            raise ValueError("PCIe effective link is not canonically derived")
        return self


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


def _read_pcie_speed(path: Path) -> float:
    raw = path.read_text(encoding="utf-8").strip()
    match = _PCIE_SPEED_RE.fullmatch(raw)
    if match is None:
        raise ValueError(f"unsupported PCIe link speed: {raw!r}")
    return float(match.group(1))


def _read_pcie_width(path: Path) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        width = int(raw)
    except ValueError as error:
        raise ValueError(f"unsupported PCIe link width: {raw!r}") from error
    if width <= 0:
        raise ValueError(f"unsupported PCIe link width: {raw!r}")
    return width


def _pcie_link_identity(path: Path) -> PCIeLinkIdentity:
    return PCIeLinkIdentity(
        bdf=path.name,
        current_speed_gtps=_read_pcie_speed(path / "current_link_speed"),
        max_speed_gtps=_read_pcie_speed(path / "max_link_speed"),
        current_width=_read_pcie_width(path / "current_link_width"),
        max_width=_read_pcie_width(path / "max_link_width"),
    )


def collect_pcie_topology(
    gpu_bdf: str,
    *,
    pci_devices_root: Path = PCI_DEVICES_ROOT,
    sys_devices_root: Path = SYS_DEVICES_ROOT,
) -> PCIeTopologyIdentity:
    """Collect the complete ordered PCIe path for one GPU endpoint BDF."""
    normalized = gpu_bdf.strip().lower()
    if _PCI_BDF_RE.fullmatch(normalized) is None:
        raise ValueError(f"invalid PCI BDF: {gpu_bdf!r}")
    endpoint = (pci_devices_root / normalized).resolve(strict=True)
    resolved_sysfs = sys_devices_root.resolve(strict=True)
    if not endpoint.is_relative_to(resolved_sysfs):
        raise ValueError("PCI device symlink escapes the sysfs device tree")
    paths = tuple(
        path
        for path in reversed((endpoint, *endpoint.parents))
        if _PCI_BDF_RE.fullmatch(path.name)
    )
    if not paths or paths[-1].name != normalized:
        raise ValueError("PCIe topology does not terminate at the GPU BDF")
    links = tuple(_pcie_link_identity(path) for path in paths)
    bottleneck = min(
        links,
        key=lambda link: link.current_speed_gtps * link.current_width,
    )
    return PCIeTopologyIdentity(
        links=links,
        bottleneck_bdf=bottleneck.bdf,
        effective_speed_gtps=bottleneck.current_speed_gtps,
        effective_width=bottleneck.current_width,
    )


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
    device: str = "cuda:0",
    *,
    torch_module: Any | None = None,
) -> RocmDeviceInfo:
    """Detect one concrete PyTorch ROCm device and its execution capacities."""
    if torch_module is None:
        import torch as torch_module

    parsed = torch_module.device(device)
    if parsed.type != "cuda":
        raise ValueError(
            f"ROCm target device must use the cuda namespace: {device}",
        )
    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    if hip_version is None or not torch_module.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    index = (
        parsed.index
        if parsed.index is not None
        else torch_module.cuda.current_device()
    )
    if index < 0 or index >= torch_module.cuda.device_count():
        raise ValueError(f"ROCm device index is out of range: {index}")
    properties = torch_module.cuda.get_device_properties(index)
    raw_gfx = getattr(properties, "gcnArchName", "") or getattr(
        properties,
        "gfx_arch_name",
        "",
    )
    gfx_target = str(raw_gfx).split(":", maxsplit=1)[0].strip().lower()
    if not gfx_target.startswith("gfx"):
        raise RuntimeError(
            f"ROCm device did not expose a concrete gfx target: {raw_gfx!r}",
        )
    raw_l2 = getattr(properties, "L2_cache_size", None)
    l2_cache_bytes = (
        int(raw_l2) if raw_l2 is not None and int(raw_l2) > 0 else None
    )
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


def pin_cuda_device(device: str, *, torch_module: Any | None = None) -> None:
    """Pin the active CUDA device before candidate or reference code runs.

    No-op for CPU targets and when CUDA is unavailable, so single-device and
    CPU-only executions are unaffected. On a multi-GPU host this fixes the
    active device so a candidate cannot direct timed work onto an idle device
    while producing its correct output elsewhere. A ``cuda:N`` index that is out
    of range is allowed to surface from ``set_device`` as a fail-closed error.
    """
    if torch_module is None:
        import torch as torch_module
    parsed = torch_module.device(device)
    if parsed.type != "cuda" or not torch_module.cuda.is_available():
        return
    index = (
        parsed.index
        if parsed.index is not None
        else torch_module.cuda.current_device()
    )
    torch_module.cuda.set_device(index)


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


def detect_rocm_version(
    *,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    which: Which = shutil.which,
    is_dir: Callable[[Path], bool] = Path.is_dir,
) -> str | None:
    """Return the installed ROCm user-space version from canonical files."""
    rocm_root = root or discover_rocm_root(
        environ=environ,
        which=which,
        is_dir=is_dir,
    )
    if rocm_root is None:
        return None
    for path in (rocm_root / ".info/version", rocm_root / ".info/version-dev"):
        try:
            version = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if version:
            return version
    return None


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
    candidates = [
        discover_rocm_root(environ=environ, which=which, is_dir=is_dir),
    ]
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
) -> Environment:
    """Collect the hardware and library information for *device*."""
    import torch

    from sol_execbench.core.data.trace import Environment

    libs: dict[str, str] = {"torch": torch.__version__}
    try:
        triton = import_module("triton")
        libs["triton"] = getattr(triton, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import torch.version as tv

        if hip_version := getattr(tv, "hip", None):
            libs["hip"] = str(hip_version)
        elif cuda_version := getattr(tv, "cuda", None):
            libs["cuda"] = str(cuda_version)
    except ImportError:
        pass
    if rocm_version := detect_rocm_version():
        libs["rocm"] = rocm_version
    isolation = "unknown"
    if os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) == "1":
        isolation = "container"
    elif os.environ.get(ENV_SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION) == "1":
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
        if getattr(getattr(torch, "version", None), "hip", None) is not None:
            return detect_rocm_device(device, torch_module=torch).gfx_target
        return torch.cuda.get_device_name(parsed_device.index)
    if parsed_device.type == "cpu":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (OSError, UnicodeError):
            pass
        return platform.processor() or platform.machine() or "CPU"
    if parsed_device.type == "mps":
        return "Apple GPU (MPS)"
    if parsed_device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return torch.xpu.get_device_name(parsed_device.index)
        except RuntimeError:
            return "Intel XPU"
    return parsed_device.type
