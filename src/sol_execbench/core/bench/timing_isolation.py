# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timing isolation audit infrastructure for ROCm profiling scripts.

This module provides four public functions for detecting and warning about
conditions that could introduce timing variability or measurement bias:

1. ``detect_concurrent_gpu_processes()`` — Detect concurrent GPU processes via rocm-smi
2. ``verify_clock_state_with_warning()`` — Verify STABLE_PEAK clock mode with context-aware logging
3. ``clear_gpu_cache_between_subprocesses()`` — Clear GPU cache at subprocess boundaries
4. ``collect_timing_environment_snapshot()`` — Record environment state for reproducibility audits

All functions follow graceful degradation principles: log warnings but don't raise
exceptions when probes fail or tools are unavailable.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

TIMING_ISOLATION_SNAPSHOT_SCHEMA_VERSION = "sol_execbench.timing_isolation_snapshot.v1"


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
    from sol_execbench.core.environment import collect_environment_snapshot

    # Collect base environment snapshot (without PyTorch for speed)
    base_snapshot = collect_environment_snapshot(collect_pytorch=False)

    # Collect timing-specific information
    gpu_processes = detect_concurrent_gpu_processes()
    clocks_locked = are_clocks_locked()

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
        "tools_available": tools_available,
        "warnings": warnings,
    }
