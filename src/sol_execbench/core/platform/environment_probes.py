# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded environment probe execution and conservative output parsing."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

from .environment_models import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    EnvironmentEvidenceStatus,
    ProbeCompletedProcess,
    ProbeRunner,
    PytorchRocmSummary,
    ToolProbeResult,
    Which,
)
from .runtime import resolve_rocm_tool, resolve_tool_path
from ..text_utils import text_tail


def probe_tool(
    tool: str,
    command: list[str],
    *,
    runner: ProbeRunner,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> ToolProbeResult:
    """Run one bounded environment probe."""

    # Environment doctor must use the same ROCm-root-aware lookup as the
    # collectors.  ROCm packages are commonly installed under a versioned
    # /opt/rocm-* root without adding every utility to PATH.
    # A caller-supplied ``which`` represents a hermetic probe environment in
    # tests and integrations; do not accidentally consult the host's /opt/rocm
    # installation in that case. Normal environment doctor use retains full
    # ROCm-root-aware discovery.
    resolved_path = (
        resolve_rocm_tool(tool, which=which)
        if which is shutil.which
        else resolve_tool_path(tool, which=which)
    )
    path = str(resolved_path) if resolved_path is not None else None
    if path is None:
        return ToolProbeResult(
            tool=tool,
            command=command,
            status=EnvironmentEvidenceStatus.UNAVAILABLE,
            timeout_seconds=timeout_seconds,
        )
    effective_command = [path, *command[1:]]
    try:
        completed = runner(effective_command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolProbeResult(
            tool=tool,
            command=effective_command,
            path=path,
            status=EnvironmentEvidenceStatus.TIMEOUT,
            stdout_tail=text_tail(exc.stdout),
            stderr_tail=text_tail(exc.stderr),
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        return ToolProbeResult(
            tool=tool,
            command=effective_command,
            path=path,
            status=EnvironmentEvidenceStatus.FAILED,
            stderr_tail=text_tail(str(exc)),
            timeout_seconds=timeout_seconds,
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return ToolProbeResult(
        tool=tool,
        command=effective_command,
        path=path,
        status=(
            EnvironmentEvidenceStatus.AVAILABLE
            if completed.returncode == 0
            else EnvironmentEvidenceStatus.FAILED
        ),
        returncode=completed.returncode,
        stdout_tail=text_tail(completed.stdout),
        stderr_tail=text_tail(completed.stderr),
        timeout_seconds=timeout_seconds,
        parsed=parse_probe_output(output),
    )


def collect_pytorch_rocm_summary() -> PytorchRocmSummary:
    """Collect PyTorch ROCm metadata without requiring PyTorch at import time."""

    try:
        import torch
    except ImportError as exc:
        return PytorchRocmSummary(available=False, error=str(exc))

    torch_version = str(getattr(torch, "__version__", ""))
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    try:
        available = bool(torch.cuda.is_available()) and hip_version is not None
        device_count = int(torch.cuda.device_count()) if available else 0
        device_name = torch.cuda.get_device_name(0) if device_count else None
        gfx_target = None
        if device_count:
            props = torch.cuda.get_device_properties(0)
            raw_arch = getattr(props, "gcnArchName", "") or getattr(
                props, "gfx_arch_name", ""
            )
            gfx_target = str(raw_arch).split(":", maxsplit=1)[0] or None
        return PytorchRocmSummary(
            available=available,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            device_count=device_count,
            device_name=device_name,
            gfx_target=gfx_target,
        )
    except (RuntimeError, AttributeError) as exc:
        return PytorchRocmSummary(
            available=False,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            error=str(exc),
        )


def run_probe(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return ProbeCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def parse_probe_output(output: str) -> dict[str, Any]:
    # ROCm 7.2 emits generic labels such as ``gfx12`` in addition to concrete
    # ISA targets.  Only the latter identify a device architecture.
    gfx_targets = sorted(set(re.findall(r"\bgfx[0-9a-fA-F]{3,}\b", output)))
    parsed: dict[str, Any] = {}
    if gfx_targets:
        parsed["gfx_targets"] = gfx_targets
    marketing_names = []
    for line in output.splitlines():
        if "Marketing Name" in line or "GPU" in line and "Name" in line:
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                marketing_names.append(value)
    if marketing_names:
        parsed["names"] = marketing_names[:8]
    return parsed
