# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROCm GPU clock locking for stable SOL ExecBench benchmark timing.

Uses ``amd-smi set -l STABLE_PEAK`` to lock clocks to a reproducible near-maximum
performance level. This is the only reliably stable method on RDNA4 hardware.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

from pydantic import ValidationError

from sol_execbench.core.platform.amd_smi import parse_performance_levels
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

logger = logging.getLogger(__name__)

VERIFY_DELAY_S = 3
AMD_SMI_FAILURE_MARKERS = (
    "unable to set performance level",
    "failed to set",
    "error:",
)


def _amd_smi_executable() -> str:
    return resolve_rocm_tool_command("amd-smi")


def _has_command_failure(
    result: subprocess.CompletedProcess, markers: tuple[str, ...]
) -> str | None:
    output_parts = []
    for value in (result.stdout, result.stderr):
        if not value:
            continue
        if isinstance(value, bytes):
            output_parts.append(value.decode(errors="replace"))
        else:
            output_parts.append(str(value))
    output = "\n".join(output_parts)
    normalized = output.lower()
    for marker in markers:
        if marker in normalized:
            return output.strip()
    return None


def _run_checked_amd_smi(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )
    failure = _has_command_failure(result, AMD_SMI_FAILURE_MARKERS)
    if failure:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result


def _query_performance_levels() -> tuple[str, ...]:
    """Return the performance level reported for every visible AMD GPU."""
    result = subprocess.run(
        [_amd_smi_executable(), "metric", "-l", "--json"],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return parse_performance_levels(result.stdout)


def _all_gpus_at_level(level: str) -> bool:
    expected = f"AMDSMI_DEV_PERF_LEVEL_{level.upper()}"
    try:
        levels = _query_performance_levels()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning("Failed to query amd-smi performance levels: %s", exc)
        return False
    return all(current == expected for current in levels)


def probe_clock_lock_available() -> bool:
    """Check read-only sudo policy coverage for every clock lifecycle command."""
    amd_smi = _amd_smi_executable()
    commands = (
        [amd_smi, "version"],
        [amd_smi, "set", "-l", "STABLE_PEAK"],
        [amd_smi, "set", "-l", "AUTO"],
    )
    try:
        return all(
            subprocess.run(
                ["sudo", "-n", "-l", "--", *command],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
            for command in commands
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def lock_clocks() -> bool:
    """Lock GPU clocks to STABLE_PEAK via ``amd-smi``.

    Uses ``sudo amd-smi set -l STABLE_PEAK`` which sets the GPU to a
    stable near-maximum performance level with clock gating disabled.
    Returns ``True`` only when the command and verification both succeed.

    On RDNA4 ``gfx1200``, STABLE_PEAK provides:
      - SCLK at ~2709-2754Mhz (within 1-2% of maximum 2780Mhz)
      - MCLK locked at 1124Mhz (zero fluctuation)
      - Clock and power gating disabled

    """
    try:
        _run_checked_amd_smi(
            [
                "sudo",
                "-n",
                _amd_smi_executable(),
                "set",
                "-l",
                "STABLE_PEAK",
            ]
        )
        logger.info("GPU clock locked to STABLE_PEAK")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning("Failed to lock clocks (STABLE_PEAK): %s", e)
        return False

    logger.info("Waiting %ss for clocks to stabilize...", VERIFY_DELAY_S)
    time.sleep(VERIFY_DELAY_S)
    if not verify_clocks():
        logger.warning("STABLE_PEAK verification failed; unlocking")
        unlock_clocks()
        return False

    return True


def verify_clocks() -> bool:
    """Return whether every visible AMD GPU is in STABLE_PEAK mode."""
    locked = _all_gpus_at_level("STABLE_PEAK")
    if not locked:
        logger.warning("Not all visible GPUs are in STABLE_PEAK mode")
    return locked


def unlock_clocks() -> bool:
    """Reset GPU clocks to auto via ``amd-smi set -l AUTO``.

    Best-effort; errors are logged but not raised. Returns whether every visible
    GPU was verified back in AUTO mode.
    """
    try:
        _run_checked_amd_smi(["sudo", "-n", _amd_smi_executable(), "set", "-l", "AUTO"])
    except Exception as e:
        logger.warning("Failed to unlock clocks: %s", e)
        return False
    if not _all_gpus_at_level("AUTO"):
        logger.warning("GPU clock reset could not be verified at AUTO")
        return False
    return True


def are_clocks_locked() -> bool:
    """Check whether clocks were locked successfully before evaluation."""
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
