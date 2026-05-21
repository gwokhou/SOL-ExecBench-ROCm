# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ROCm GPU clock locking for stable SOL ExecBench benchmark timing."""

from __future__ import annotations

import logging
import os
import subprocess
import time

from .config.device_config import get_clock_preset

logger = logging.getLogger(__name__)

VERIFY_DELAY_S = 3


def probe_clock_lock_available() -> bool:
    """Probe whether ROCm clock tooling is available via passwordless sudo."""
    try:
        probe = subprocess.run(
            ["sudo", "-n", "rocm-smi", "--showclocks"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0


def lock_clocks(device_name: str) -> bool:
    """Lock ROCm SCLK and MCLK DPM levels for the given device.

    DPM levels can be overridden with ``SOL_EXECBENCH_SCLK_LEVEL`` and
    ``SOL_EXECBENCH_MCLK_LEVEL``. The function returns ``True`` only when the
    lock commands and verification both succeed.
    """
    preset = get_clock_preset(device_name)
    sclk_level_str = os.environ.get("SOL_EXECBENCH_SCLK_LEVEL")
    mclk_level_str = os.environ.get("SOL_EXECBENCH_MCLK_LEVEL")

    sclk_level = (
        int(sclk_level_str)
        if sclk_level_str
        else (preset.sclk_level if preset else None)
    )
    mclk_level = (
        int(mclk_level_str)
        if mclk_level_str
        else (preset.mclk_level if preset else None)
    )

    if sclk_level is None:
        logger.warning(
            "No ROCm SCLK preset for '%s' and SOL_EXECBENCH_SCLK_LEVEL not set",
            device_name,
        )
        return False
    if mclk_level is None:
        logger.warning(
            "No ROCm MCLK preset for '%s' and SOL_EXECBENCH_MCLK_LEVEL not set",
            device_name,
        )
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "rocm-smi", "--setperflevel", "manual"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["sudo", "-n", "rocm-smi", "--setsclk", str(sclk_level)],
            check=True,
            capture_output=True,
        )
        logger.info("ROCm SCLK locked to DPM level %s", sclk_level)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Failed to lock ROCm SCLK: %s", e)
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "rocm-smi", "--setmclk", str(mclk_level)],
            check=True,
            capture_output=True,
        )
        logger.info("ROCm MCLK locked to DPM level %s", mclk_level)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Failed to lock ROCm MCLK: %s", e)
        unlock_clocks()
        return False

    logger.info("Waiting %ss for ROCm clocks to stabilize...", VERIFY_DELAY_S)
    time.sleep(VERIFY_DELAY_S)
    if not verify_clocks(sclk_level, mclk_level):
        logger.warning("ROCm clock verification failed after locking; unlocking")
        unlock_clocks()
        return False

    return True


def _level_is_active(stdout: str, clock_name: str, expected_level: int) -> bool:
    expected = str(expected_level)
    in_section = False
    for line in stdout.splitlines():
        normalized = line.lower()
        if "clock level" in normalized:
            in_section = clock_name in normalized
            continue
        if not in_section:
            continue
        if "*" in line and f"{expected}:" in line:
            return True
        if "current" in normalized and expected in line:
            return True
    return False


def verify_clocks(expected_sclk_level: int, expected_mclk_level: int) -> bool:
    """Verify current ROCm SCLK and MCLK DPM levels via ``rocm-smi``."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showclocks"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("rocm-smi clock query failed: %s", result.stderr.strip())
            return False
    except FileNotFoundError:
        logger.warning("rocm-smi not found; cannot verify ROCm clocks")
        return False

    stdout = result.stdout.strip()
    if not stdout:
        logger.warning("rocm-smi returned no clock data")
        return False

    sclk_ok = _level_is_active(stdout, "sclk", expected_sclk_level)
    mclk_ok = _level_is_active(stdout, "mclk", expected_mclk_level)
    if not sclk_ok:
        logger.warning("ROCm SCLK level %s is not active", expected_sclk_level)
    if not mclk_ok:
        logger.warning("ROCm MCLK level %s is not active", expected_mclk_level)
    return sclk_ok and mclk_ok


def unlock_clocks() -> None:
    """Reset ROCm clocks. Best-effort; errors are logged but not raised."""
    for command in (
        ["sudo", "-n", "rocm-smi", "--resetclocks"],
        ["sudo", "-n", "rocm-smi", "--setperflevel", "auto"],
    ):
        try:
            subprocess.run(command, capture_output=True)
        except Exception as e:
            logger.warning("Failed to run %s: %s", " ".join(command), e)


def are_clocks_locked() -> bool:
    """Check whether clocks were locked successfully before evaluation."""
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
