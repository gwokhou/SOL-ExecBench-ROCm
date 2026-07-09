# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timing isolation audit infrastructure for ROCm profiling scripts.

This module provides five public functions for detecting and warning about
conditions that could introduce timing variability or measurement bias:

1. ``detect_concurrent_gpu_processes()`` — Detect concurrent GPU processes via rocm-smi
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
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

TIMING_ISOLATION_SNAPSHOT_SCHEMA_VERSION = "sol_execbench.timing_isolation_snapshot.v1"
GPU_ISOLATION_SCHEMA_VERSION = "sol_execbench.gpu_device_isolation.v1"


def detect_concurrent_gpu_processes() -> list[dict[str, Any]]:
    """Detect concurrent GPU processes via ``rocm-smi --showpids``.

    Returns a list of process dicts with keys: ``pid``, ``device``, ``name``.
    Returns an empty list if no processes are running, on timeout, or on error.

    This function uses a 5-second timeout and degrades gracefully — it logs warnings
    but never raises exceptions, following the pitfall avoidance guidance from RESEARCH.md.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.warning("rocm-smi not found; cannot detect concurrent GPU processes")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("rocm-smi --showpids timed out after 5 seconds")
        return []

    stdout = result.stdout or ""

    # Check for "No KFD PIDs" marker
    if "No KFD PIDs" in stdout:
        return []

    # Parse ROCm SMI output to extract process information
    processes = []
    lines = stdout.splitlines()

    # Track current device
    current_device = "unknown"

    for line in lines:
        line = line.strip()

        # Detect GPU device lines
        if "GPU" in line and ("0000:" in line or ":" in line):
            parts = line.split()
            for part in parts:
                if ":" in part and part.count(":") >= 2:
                    current_device = part
                    break

        # Look for KFD PID lines
        if "KFD PID" in line and "Name" in line:
            # Extract PID and name from lines like:
            # "KFD PID                     12345  Name              python"
            parts = line.split()
            try:
                # Find all numeric values (potential PIDs)
                for i, part in enumerate(parts):
                    if part.isdigit():
                        pid = int(part)

                        # Find the name (typically after "Name")
                        name = "unknown"
                        if "Name" in parts:
                            name_idx = parts.index("Name") + 1
                            if name_idx < len(parts):
                                name = parts[name_idx]

                        processes.append(
                            {
                                "pid": pid,
                                "device": current_device,
                                "name": name,
                            }
                        )
                        break  # Only take first PID from line
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    return processes


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
    """Detect GPU count via ``rocm-smi --showid``.

    Returns 0 on error or when rocm-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0

    gpu_ids: set[str] = set()
    for line in (result.stdout or "").splitlines():
        match = re.match(r"^\s*GPU\[(\d+)\]\s*:", line)
        if match:
            gpu_ids.add(match.group(1))
    return len(gpu_ids)


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
        warnings.append("gpu_count_unknown: rocm-smi unavailable or returned no GPUs")
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
