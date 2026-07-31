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
import subprocess
from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationError

from sol_execbench.core.data.base_model import CurrentSchemaModel
from sol_execbench.core.integrity.schema_versions import (
    GPU_DEVICE_ISOLATION_SCHEMA_VERSION,
)
from sol_execbench.core.platform.amd_smi import parse_gpu_count, parse_processes
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

logger = logging.getLogger(__name__)


class GPUDeviceIsolation(CurrentSchemaModel):
    """Current GPU isolation observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    current_schema_version = GPU_DEVICE_ISOLATION_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.gpu_device_isolation.v1"] = (
        "sol_execbench.gpu_device_isolation.v1"
    )
    isolated: bool
    gpu_count: int = Field(ge=0)
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


def detect_concurrent_gpu_processes() -> list[dict[str, Any]]:
    """Detect concurrent GPU processes via ``amd-smi process --json``.

    Returns a list of process dicts with keys: ``pid``, ``device``, ``name``.
    Returns an empty list if no processes are running, on timeout, or on error.

    This function uses a 5-second timeout and degrades gracefully — it logs warnings
    but never raises exceptions, following the pitfall avoidance guidance from RESEARCH.md.
    """
    try:
        raw = _run_amd_smi_json("process")
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
        return parse_processes(raw)
    except (ValidationError, ValueError) as exc:
        logger.warning("amd-smi process payload was invalid: %s", exc)
        return []


def verify_clock_state_with_warning(context: str = "batch_start") -> bool:
    """Verify GPU clock state is STABLE_PEAK with context-aware logging.

    Wraps ``verify_clocks()`` from ``clock_lock`` module and logs informational
    or warning messages depending on verification result.

    Args:
        context: Context string for log messages (e.g., ``"batch_start"``, ``"problem_42"``)

    Returns:
        ``True`` if clocks are in STABLE_PEAK mode, ``False`` otherwise

    """
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
    """Detect GPU count via ``amd-smi list --json``.

    Returns 0 on error or when amd-smi is unavailable.
    """
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


def validate_gpu_device_isolation(
    *,
    gpu_device: int | None = None,
) -> dict[str, Any]:
    """Validate GPU device isolation for timing-sensitive workloads.

    Checks whether the process has adequate GPU device isolation by examining
    ``ROCR_VISIBLE_DEVICES`` and the total GPU count. Optionally sets
    ``ROCR_VISIBLE_DEVICES`` when ``gpu_device`` is provided.

    The caller decides whether to warn or abort based on the ``isolated`` result.

    Args:
        gpu_device: If provided, set ``ROCR_VISIBLE_DEVICES`` to this device index
            for the current process before checking.

    Returns:
        Dict with keys:
        - ``schema_version``: Isolation check schema identifier
        - ``isolated``: Whether the process has adequate GPU isolation
        - ``gpu_count``: Total GPU count detected (0 if unknown)
        - ``rocr_visible_devices``: Current ``ROCR_VISIBLE_DEVICES`` value or None
        - ``gpu_device_set``: Whether a specific device was requested and set
        - ``warnings``: List of non-fatal warnings

    """
    warnings: list[str] = []

    if gpu_device is not None:
        os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_device)
        logger.info(
            "Set ROCR_VISIBLE_DEVICES=%d for GPU device isolation",
            gpu_device,
        )

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    gpu_count = _detect_gpu_count()

    if gpu_count == 0:
        warnings.append(
            "gpu_count_unknown: amd-smi unavailable or returned no GPUs",
        )
    elif gpu_count > 1 and rocr_visible is None:
        warnings.append(
            f"multi_gpu_no_restriction: {gpu_count} GPUs detected but "
            "ROCR_VISIBLE_DEVICES not set — timing may be affected by "
            "cross-device interference",
        )

    isolated = gpu_count <= 1 or rocr_visible is not None

    result = {
        "schema_version": GPU_DEVICE_ISOLATION_SCHEMA_VERSION,
        "isolated": isolated,
        "gpu_count": gpu_count,
        "rocr_visible_devices": rocr_visible,
        "gpu_device_set": gpu_device is not None,
        "warnings": warnings,
    }
    return GPUDeviceIsolation.model_validate(result).model_dump(mode="json")
