# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm build configuration helpers for staged native solutions."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from typing import Any

from sol_execbench.core.data.solution import CompileOptions, SupportedHardware
from sol_execbench.core.text_utils import ordered_unique


def first_gfx_target(lines: list[str]) -> str | None:
    """Return the first concrete AMD gfx target from command output lines."""
    for line in lines:
        match = re.search(r"\bgfx[0-9a-fA-F]+\b", line)
        if match and match.group(0) != "gfx000":
            return match.group(0)
    return None


def get_local_gfx(
    *,
    check_output: Callable[..., str] = subprocess.check_output,
) -> str | None:
    """Detect the local AMD GPU gfx target using ROCm tooling."""
    try:
        out = check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        target = first_gfx_target(out.splitlines())
        if target:
            return target
    except Exception:
        pass

    try:
        out = check_output(
            ["rocminfo"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return first_gfx_target(out.splitlines())
    except Exception:
        return None


def gfx_to_offload_arch(gfx: str) -> str:
    """Convert an AMD gfx target string to a HIP offload architecture flag."""
    return f"--offload-arch={gfx}"


def inject_offload_arch_flags(
    sol_dict: dict[str, Any],
    *,
    local_gfx_getter: Callable[[], str | None] = get_local_gfx,
) -> dict[str, Any]:
    """Auto-inject HIP offload architecture flags when none are explicit."""
    spec = sol_dict["spec"]
    compile_options = dict(spec.get("compile_options") or {})
    if "hip_cflags" not in compile_options:
        # Preserve the CompileOptions default (e.g. -O3) instead of clobbering
        # it with an empty list when no flags were specified.
        compile_options["hip_cflags"] = list(CompileOptions().hip_cflags)
    hip_cflags = list(compile_options["hip_cflags"])

    if any(
        flag.startswith(("--offload-arch", "-offload-arch", "--amdgpu-target"))
        for flag in hip_cflags
    ):
        return sol_dict

    offload_arches: list[str] = []
    target_hw = set(spec.get("target_hardware", []))

    for hardware in SupportedHardware:
        value = hardware.value
        if value != SupportedHardware.LOCAL.value and value in target_hw:
            offload_arches.append(value)

    if SupportedHardware.LOCAL.value in target_hw:
        local_gfx = local_gfx_getter()
        if local_gfx:
            offload_arches.append(local_gfx)

    unique = ordered_unique(offload_arches)
    if unique:
        compile_options["hip_cflags"] = [
            gfx_to_offload_arch(gfx) for gfx in unique
        ] + hip_cflags
        spec["compile_options"] = compile_options
        sol_dict["spec"] = spec

    return sol_dict
