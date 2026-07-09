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
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

VERIFY_DELAY_S = 3
AMD_SMI_FAILURE_MARKERS = (
    "unable to set performance level",
    "failed to set",
    "error:",
)


def _rocm_smi_executable() -> str:
    return shutil.which("rocm-smi") or "rocm-smi"


def _amd_smi_executable() -> str:
    return shutil.which("amd-smi") or "amd-smi"


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
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    failure = _has_command_failure(result, AMD_SMI_FAILURE_MARKERS)
    if failure:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result


def probe_clock_lock_available() -> bool:
    """Probe whether ``amd-smi`` is available via passwordless sudo."""
    try:
        probe = subprocess.run(
            ["sudo", "-n", _amd_smi_executable(), "version"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0


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
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
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
    """Verify that the GPU is in STABLE_PEAK mode via ``rocm-smi --showperflevel``."""
    try:
        result = subprocess.run(
            [_rocm_smi_executable(), "--showperflevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "rocm-smi perf level query failed: %s", result.stderr.strip()
            )
            return False
    except FileNotFoundError:
        logger.warning("rocm-smi not found; cannot verify clock state")
        return False

    stdout = (result.stdout or "").lower()
    if not stdout.strip():
        logger.warning("rocm-smi returned no data")
        return False

    if "stable_peak" in stdout:
        return True

    logger.warning("GPU is not in STABLE_PEAK mode")
    return False


def unlock_clocks() -> None:
    """Reset GPU clocks to auto via ``amd-smi set -l AUTO``.

    Best-effort; errors are logged but not raised.
    """
    try:
        subprocess.run(
            ["sudo", "-n", _amd_smi_executable(), "set", "-l", "AUTO"],
            capture_output=True,
        )
    except Exception as e:
        logger.warning("Failed to unlock clocks: %s", e)


def are_clocks_locked() -> bool:
    """Check whether clocks were locked successfully before evaluation."""
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
