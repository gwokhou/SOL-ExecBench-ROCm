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
from dataclasses import dataclass, field

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


@dataclass
class ClockLockLease:
    """Owned STABLE_PEAK lease with idempotent, exception-aware release."""

    locked: bool
    acquired: bool
    _released: bool = field(default=False, init=False, repr=False, compare=False)
    _previous_environment: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _environment_published: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _detached: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def active(self) -> bool:
        """Return whether this object still owns an unreleased GPU policy."""
        return self.acquired and not self._released and not self._detached

    @property
    def released(self) -> bool:
        """Return whether release completed or no owned policy required it."""
        return self._released or not self.acquired

    @property
    def detached(self) -> bool:
        """Return whether release responsibility was explicitly transferred."""
        return self._detached

    def release(self) -> bool:
        """Release an owned lock, retaining ownership when reset fails."""
        if self._detached:
            raise RuntimeError("cannot release a detached GPU clock lease")
        if self._released or not self.acquired:
            self._released = True
            return True
        if not unlock_clocks():
            return False
        self._released = True
        return True

    def detach(self) -> None:
        """Transfer or intentionally retain release responsibility."""
        if self._released:
            raise RuntimeError("cannot detach a released GPU clock lease")
        self._detached = True

    def __del__(self) -> None:
        try:
            if self.active:
                logger.error(
                    "gpu_clock_lease_leaked: owned STABLE_PEAK lease was "
                    "garbage-collected without release or detach"
                )
        except BaseException:
            # Destructors must never turn observability into shutdown failure.
            pass

    def __enter__(self) -> "ClockLockLease":
        return self

    def publish_environment(self) -> None:
        """Publish verified state while retaining the previous environment."""
        if self._environment_published:
            return
        self._previous_environment = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED")
        os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1" if self.locked else "0"
        self._environment_published = True

    def restore_environment(self) -> None:
        """Restore the environment captured by :meth:`publish_environment`."""
        if not self._environment_published:
            return
        if self._previous_environment is None:
            os.environ.pop("SOL_EXECBENCH_CLOCKS_LOCKED", None)
        else:
            os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = self._previous_environment
        self._environment_published = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        if self.release():
            return
        message = "failed to reset and verify every GPU at AUTO"
        if exc_value is not None:
            exc_value.add_note(message)
            return
        raise RuntimeError(message)


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


def _observed_performance_levels() -> tuple[str, ...] | None:
    try:
        return _query_performance_levels()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning("Failed to query amd-smi performance levels: %s", exc)
        return None


def _levels_are(levels: tuple[str, ...], level: str) -> bool:
    expected = f"AMDSMI_DEV_PERF_LEVEL_{level.upper()}"
    return all(current == expected for current in levels)


def _all_gpus_at_level(level: str) -> bool:
    levels = _observed_performance_levels()
    return levels is not None and _levels_are(levels, level)


def _reset_after_failed_lock() -> None:
    """Best-effort reset after a lock command may have changed GPU policy."""
    if not unlock_clocks():
        logger.error("Failed to restore GPU clocks to AUTO after lock failure")


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


def acquire_clock_lock() -> ClockLockLease:
    """Request STABLE_PEAK and report whether this process changed the policy.

    Uses ``sudo amd-smi set -l STABLE_PEAK`` which sets the GPU to a
    stable near-maximum performance level with clock gating disabled. Any
    unsuccessful exit after the mutating command starts attempts to restore
    AUTO, including interruption during stabilization or verification.

    On RDNA4 ``gfx1200``, STABLE_PEAK provides:
      - SCLK at ~2709-2754Mhz (within 1-2% of maximum 2780Mhz)
      - MCLK locked at 1124Mhz (zero fluctuation)
      - Clock and power gating disabled

    """
    initial_levels = _observed_performance_levels()
    if initial_levels is None:
        return ClockLockLease(locked=False, acquired=False)
    if _levels_are(initial_levels, "STABLE_PEAK"):
        logger.info("GPU clocks are already at STABLE_PEAK; preserving external lock")
        return ClockLockLease(locked=True, acquired=False)
    if not _levels_are(initial_levels, "AUTO"):
        logger.warning(
            "Refusing to replace non-AUTO GPU performance levels: %s",
            initial_levels,
        )
        return ClockLockLease(locked=False, acquired=False)

    command_started = False
    try:
        command_started = True
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
        logger.info("Waiting %ss for clocks to stabilize...", VERIFY_DELAY_S)
        time.sleep(VERIFY_DELAY_S)
        if not verify_clocks():
            logger.warning("STABLE_PEAK verification failed; unlocking")
            _reset_after_failed_lock()
            return ClockLockLease(locked=False, acquired=False)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        logger.warning("Failed to lock clocks (STABLE_PEAK): %s", exc)
        if command_started:
            _reset_after_failed_lock()
        return ClockLockLease(locked=False, acquired=False)
    except BaseException:
        if command_started:
            _reset_after_failed_lock()
        raise

    return ClockLockLease(locked=True, acquired=True)


def lock_clocks() -> bool:
    """Compatibility wrapper returning whether GPUs are at STABLE_PEAK."""
    lease = acquire_clock_lock()
    if lease.acquired:
        lease.detach()
    return lease.locked


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
