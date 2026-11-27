# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timing isolation audit infrastructure for ROCm profiling scripts.

This module provides three functions for detecting and warning about
conditions that could introduce timing variability or measurement bias:

1. ``detect_concurrent_gpu_processes()`` — Detect concurrent GPU processes via amd-smi
2. ``verify_clock_state_with_warning()`` — Verify STABLE_PEAK clock mode with context-aware logging
3. ``validate_gpu_device_isolation()`` — Validate GPU device isolation for timing-sensitive workloads

All functions follow graceful degradation principles: log warnings but don't raise
exceptions when probes fail or tools are unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import TypedDict

from pydantic import Field, ValidationError

from sol_execbench.core.bench.gpu_lock_paths import (
    GPU_LOCK_UUID_ENV,
    UUID_PATTERN,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.platform.amd_smi import (
    AMDSMIProcessPayload,
    parse_gpu_count,
    parse_gpu_identity,
    parse_processes,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

logger = logging.getLogger(__name__)


class GPUDeviceIsolation(FrozenArtifactModel):
    """GPU isolation observation nested in a versioned calibration."""

    isolated: bool
    gpu_count: int = Field(ge=0)
    rocr_visible_devices: str | None
    gpu_device_set: bool
    warnings: list[str]


class GPUDeviceIsolationSnapshot(TypedDict):
    """JSON-compatible public view of one isolation observation."""

    isolated: bool
    gpu_count: int
    rocr_visible_devices: str | None
    gpu_device_set: bool
    warnings: list[str]


def _run_amd_smi_json(*arguments: str) -> str:
    result = subprocess.run(
        [resolve_rocm_tool_command("amd-smi"), *arguments, "--json"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def _configured_target_uuid() -> str | None:
    raw = os.environ.get(GPU_LOCK_UUID_ENV, "").strip().lower()
    if raw and UUID_PATTERN.fullmatch(raw) is not None:
        return raw
    return None


def _resolve_target_gpu_index(gpu_uuid: str) -> int | str | None:
    try:
        list_raw = _run_amd_smi_json("list")
        return parse_gpu_identity(list_raw, gpu_uuid=gpu_uuid).gpu
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValidationError,
        ValueError,
    ):
        return None


def detect_concurrent_gpu_processes() -> list[dict[str, int | str]]:
    """Detect concurrent foreign GPU processes via ``amd-smi process --json``.

    Excludes ``os.getpid()`` and scopes queries to ``GPU_LOCK_UUID`` when set.
    Returns an empty list if no foreign processes are running, on timeout, or on error.
    """
    target_uuid = _configured_target_uuid()
    query = ("process", "--gpu", target_uuid) if target_uuid else ("process",)
    try:
        raw = _run_amd_smi_json(*query)
    except FileNotFoundError:
        logger.warning(
            "amd-smi not found; cannot detect concurrent GPU processes",
        )
        return []
    except subprocess.TimeoutExpired:
        logger.warning("amd-smi process timed out after 5 seconds")
        return []
    except subprocess.CalledProcessError as exc:
        logger.warning("amd-smi process query failed: %s", exc)
        return []
    try:
        gpu_index: int | str | None = None
        if (
            target_uuid is not None
            and len(AMDSMIProcessPayload.model_validate_json(raw).root) > 1
        ):
            gpu_index = _resolve_target_gpu_index(target_uuid)
        processes = parse_processes(raw, gpu_index=gpu_index)
    except (ValidationError, ValueError) as exc:
        logger.warning("amd-smi process payload was invalid: %s", exc)
        return []
    self_pid = os.getpid()
    return [proc for proc in processes if proc["pid"] != self_pid]


def verify_clock_state_with_warning(context: str = "batch_start") -> bool:
    """Verify GPU clock state is STABLE_PEAK with context-aware logging."""
    from sol_execbench.core.bench.clock_lock import verify_clocks

    clocks_locked = verify_clocks()

    if clocks_locked:
        logger.info(
            "Clock state verified at %s: STABLE_PEAK mode confirmed",
            context,
        )
    else:
        logger.warning(
            "Clock state verification failed at %s: GPU not in STABLE_PEAK mode. "
            "Timing measurements may be unstable.",
            context,
        )

    return clocks_locked


def _detect_gpu_count() -> int:
    """Detect GPU count via ``amd-smi list --json``."""
    try:
        raw = _run_amd_smi_json("list")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return 0
    try:
        return parse_gpu_count(raw)
    except (ValidationError, ValueError):
        return 0


def _is_single_rocr_selector(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return (
        stripped.isdigit()
        or re.fullmatch(ROCR_UUID_PATTERN, stripped) is not None
    )


def validate_gpu_device_isolation(
    *,
    gpu_device: int | str | None = None,
) -> GPUDeviceIsolationSnapshot:
    """Validate GPU device isolation for timing-sensitive workloads."""
    warnings: list[str] = []

    if gpu_device is not None:
        selector = (
            str(gpu_device)
            if isinstance(gpu_device, int)
            else gpu_device.strip()
        )
        os.environ["ROCR_VISIBLE_DEVICES"] = selector
        logger.info(
            "Set ROCR_VISIBLE_DEVICES=%s for GPU device isolation",
            selector,
        )

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    gpu_count = _detect_gpu_count()
    single_selector = _is_single_rocr_selector(rocr_visible)

    if gpu_count == 0:
        warnings.append(
            "gpu_count_unknown: amd-smi unavailable or returned no GPUs",
        )
    elif gpu_count > 1 and not single_selector:
        warnings.append(
            f"multi_gpu_no_restriction: {gpu_count} GPUs detected but "
            "ROCR_VISIBLE_DEVICES does not select a single GPU — timing may "
            "be affected by cross-device interference",
        )

    isolated = gpu_count <= 1 or single_selector

    observation = GPUDeviceIsolation(
        isolated=isolated,
        gpu_count=gpu_count,
        rocr_visible_devices=rocr_visible,
        gpu_device_set=gpu_device is not None,
        warnings=warnings,
    )
    return GPUDeviceIsolationSnapshot(
        isolated=observation.isolated,
        gpu_count=observation.gpu_count,
        rocr_visible_devices=observation.rocr_visible_devices,
        gpu_device_set=observation.gpu_device_set,
        warnings=observation.warnings,
    )
