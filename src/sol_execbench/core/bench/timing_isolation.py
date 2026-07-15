# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timing isolation audit infrastructure for ROCm profiling scripts.

This module provides five public functions for detecting and warning about
conditions that could introduce timing variability or measurement bias:

1. ``detect_concurrent_gpu_processes()`` — Detect concurrent GPU processes via amd-smi
2. ``verify_clock_state_with_warning()`` — Verify STABLE_PEAK clock mode with context-aware logging
3. ``clear_gpu_cache_between_subprocesses()`` — Clear GPU cache at subprocess boundaries
4. ``collect_timing_environment_snapshot()`` — Record environment state for reproducibility audits
5. ``validate_gpu_device_isolation()`` — Validate GPU device isolation for timing-sensitive workloads

All functions follow graceful degradation principles: log warnings but don't raise
exceptions when probes fail or tools are unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from sol_execbench.core.platform.amd_smi import parse_gpu_count, parse_processes
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

logger = logging.getLogger(__name__)

TIMING_ISOLATION_SNAPSHOT_SCHEMA_VERSION = "sol_execbench.timing_isolation_snapshot.v1"
GPU_ISOLATION_SCHEMA_VERSION = "sol_execbench.gpu_device_isolation.v1"


def _run_amd_smi_json(*arguments: str) -> str:
    result = subprocess.run(
        [resolve_rocm_tool_command("amd-smi"), *arguments, "--json"],
        capture_output=True,
        text=True,
        timeout=5,
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
        logger.warning("amd-smi not found; cannot detect concurrent GPU processes")
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


def clear_gpu_cache_between_subprocesses() -> None:
    """Clear GPU cache at subprocess boundaries via ``torch.cuda.empty_cache()``.

    Imports torch inside the function to avoid requiring it at module level.
    Logs a debug message on success and warnings on any exceptions.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared at subprocess boundary")
        else:
            logger.debug(
                "torch.cuda.is_available() returned False; skipping cache clear"
            )
    except ImportError:
        logger.debug("torch not available; skipping GPU cache clear")
    except Exception as e:
        logger.warning("Failed to clear GPU cache: %s", e)


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
        logger.info("Set ROCR_VISIBLE_DEVICES=%d for GPU device isolation", gpu_device)

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    gpu_count = _detect_gpu_count()

    if gpu_count == 0:
        warnings.append("gpu_count_unknown: amd-smi unavailable or returned no GPUs")
    elif gpu_count > 1 and rocr_visible is None:
        warnings.append(
            f"multi_gpu_no_restriction: {gpu_count} GPUs detected but "
            "ROCR_VISIBLE_DEVICES not set — timing may be affected by "
            "cross-device interference"
        )

    isolated = gpu_count <= 1 or rocr_visible is not None

    return {
        "schema_version": GPU_ISOLATION_SCHEMA_VERSION,
        "isolated": isolated,
        "gpu_count": gpu_count,
        "rocr_visible_devices": rocr_visible,
        "gpu_device_set": gpu_device is not None,
        "warnings": warnings,
    }


def collect_timing_environment_snapshot() -> dict[str, Any]:
    """Collect timing environment snapshot for reproducibility audits.

    Returns a dict with keys:
    - ``schema_version``: Snapshot schema version identifier
    - ``generated_at``: UTC timestamp in ISO format
    - ``gpu_processes``: List of concurrent GPU processes (from ``detect_concurrent_gpu_processes``)
    - ``clocks_locked``: Whether clocks are in STABLE_PEAK mode (from ``are_clocks_locked``)
    - ``tools_available``: Map of tool name to availability status
    - ``warnings``: List of non-fatal collection warnings

    The snapshot is designed for JSON serialization in batch summary sidecars.
    """
    from sol_execbench.core.bench.clock_lock import are_clocks_locked
    from sol_execbench.core.platform.environment import collect_environment_snapshot

    # Collect base environment snapshot (without PyTorch for speed)
    base_snapshot = collect_environment_snapshot(collect_pytorch=False)

    # Collect timing-specific information
    gpu_processes = detect_concurrent_gpu_processes()
    clocks_locked = are_clocks_locked()
    gpu_isolation = validate_gpu_device_isolation()

    # Build tools_available map from base snapshot
    tools_available = {}
    for tool_name, tool_result in base_snapshot.tools.items():
        tools_available[tool_name] = tool_result.status.value

    # Collect warnings
    warnings = list(base_snapshot.warnings)
    if gpu_processes:
        warnings.append(
            f"concurrent_gpu_processes: {len(gpu_processes)} process(es) detected"
        )
    if not clocks_locked:
        warnings.append("clocks_not_locked: GPU not in STABLE_PEAK mode")

    return {
        "schema_version": TIMING_ISOLATION_SNAPSHOT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "gpu_processes": gpu_processes,
        "clocks_locked": clocks_locked,
        "gpu_isolation": gpu_isolation,
        "tools_available": tools_available,
        "warnings": warnings,
    }
