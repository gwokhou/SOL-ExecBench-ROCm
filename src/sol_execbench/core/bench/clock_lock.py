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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import ValidationError

from sol_execbench.core.bench.clock_diagnostics import log_clock_command
from sol_execbench.core.platform.amd_smi import parse_performance_levels
from sol_execbench.core.platform.clock_commands import (
    ClockLevel,
    clock_command,
    clock_executable,
    host_clock_executable,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_CLOCKS_LOCKED,
)
from sol_execbench.core.process.termination import defer_termination

logger = logging.getLogger(__name__)

VERIFY_DELAY_S = 3
AMD_SMI_FAILURE_MARKERS = (
    "unable to set performance level",
    "failed to set",
    "error:",
)


class ClockSetupError(RuntimeError):
    """STABLE_PEAK setup failed; continuation requires verified rollback."""


class ClockRestoreError(RuntimeError):
    """AUTO restoration is unverified; never continue as an unlocked run."""


@dataclass(slots=True, kw_only=True)
class ClockLockLease:
    """Owned STABLE_PEAK lease with idempotent, exception-aware release."""

    locked: bool
    acquired: bool
    restore: Callable[[], bool] | None = field(default=None, repr=False)
    release_blocker: str | None = None
    _released: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _previous_environment: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _environment_published: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _detached: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

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
        if self.release_blocker is not None:
            return False
        reset = self.restore or unlock_clocks
        with defer_termination():
            if not reset():
                return False
        self._released = True
        return True

    def detach(self) -> None:
        """Transfer or intentionally retain release responsibility."""
        if self._released:
            raise RuntimeError("cannot detach a released GPU clock lease")
        self._detached = True

    def __del__(self) -> None:
        """Report leaked active leases without raising during finalization."""
        try:
            if self.active:
                logger.error(
                    "gpu_clock_lease_leaked: owned STABLE_PEAK lease was "
                    "garbage-collected without release or detach",
                )
        except BaseException:  # noqa: BLE001,S110 -- destructor must not escape
            # Destructors must never turn observability into shutdown failure.
            pass

    def __enter__(self) -> Self:
        """Return this lease for context-managed use."""
        return self

    def publish_environment(self) -> None:
        """Publish verified state while retaining the previous environment."""
        if self._environment_published:
            return
        self._previous_environment = os.environ.get(
            ENV_SOL_EXECBENCH_CLOCKS_LOCKED,
        )
        os.environ[ENV_SOL_EXECBENCH_CLOCKS_LOCKED] = (
            "1" if self.locked else "0"
        )
        self._environment_published = True

    def restore_environment(self) -> None:
        """Restore the environment captured by :meth:`publish_environment`."""
        if not self._environment_published:
            return
        if self._previous_environment is None:
            os.environ.pop(ENV_SOL_EXECBENCH_CLOCKS_LOCKED, None)
        else:
            os.environ[ENV_SOL_EXECBENCH_CLOCKS_LOCKED] = (
                self._previous_environment
            )
        self._environment_published = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Release the lease while preserving an active body exception."""
        if self.release():
            return
        message = "failed to reset and verify owned GPU policy at AUTO"
        if exc_value is not None:
            exc_value.add_note(message)
            return
        raise RuntimeError(message)


class ClockLockAcquirer(Protocol):
    """Injectable boundary for acquiring one owned GPU clock lease."""

    def __call__(self) -> ClockLockLease:
        """Acquire and return a clock-lock lease."""
        ...


def _amd_smi_executable(gpu_uuid: str | None = None) -> str:
    return (
        host_clock_executable() if gpu_uuid is not None else clock_executable()
    )


def _has_command_failure(
    result: subprocess.CompletedProcess[str],
    markers: tuple[str, ...],
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


def _run_checked_amd_smi(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        log_clock_command(
            command,
            error.returncode
            if isinstance(error, subprocess.CalledProcessError)
            else None,
            error.output,
            error.stderr,
            error=type(error).__name__,
        )
        raise
    except OSError as error:
        log_clock_command(command, None, None, None, error=type(error).__name__)
        raise
    failure = _has_command_failure(result, AMD_SMI_FAILURE_MARKERS)
    log_clock_command(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        error="output_failure_marker" if failure else None,
    )
    if failure:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _query_performance_levels(gpu_uuid: str | None = None) -> tuple[str, ...]:
    """Return policies for the selected UUID, or all visible AMD GPUs."""
    command = [_amd_smi_executable(gpu_uuid), "metric", "-l", "--json"]
    if gpu_uuid is not None:
        command.extend(("--gpu", gpu_uuid))
    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return parse_performance_levels(result.stdout)


def observe_clock_levels(gpu_uuid: str | None = None) -> tuple[str, ...] | None:
    """Read the selected GPU policies without changing device state."""
    try:
        return (
            _query_performance_levels()
            if gpu_uuid is None
            else _query_performance_levels(gpu_uuid)
        )
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
    return bool(levels) and all(current == expected for current in levels)


def _all_gpus_at_level(level: str, gpu_uuid: str | None = None) -> bool:
    levels = (
        observe_clock_levels()
        if gpu_uuid is None
        else observe_clock_levels(gpu_uuid)
    )
    return (
        levels is not None
        and (gpu_uuid is None or len(levels) == 1)
        and _levels_are(levels, level)
    )


def _reset_after_failed_lock() -> None:
    """Refuse continuation when a possibly changed policy cannot be restored."""
    if not unlock_clocks():
        raise ClockRestoreError("clock_restore_unverified: AUTO reset failed")


def set_peak_clocks(gpu_uuid: str | None = None) -> None:
    """Set and verify STABLE_PEAK; the caller must already own restoration."""
    try:
        _run_checked_amd_smi(
            [
                "sudo",
                "-n",
                *clock_command(
                    _amd_smi_executable(gpu_uuid), ClockLevel.PEAK, gpu_uuid
                ),
            ]
        )
        time.sleep(VERIFY_DELAY_S)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as error:
        raise ClockSetupError(
            "clock_setup_failed: STABLE_PEAK command failed"
        ) from error
    verified = verify_clocks() if gpu_uuid is None else verify_clocks(gpu_uuid)
    if not verified:
        raise ClockSetupError("clock_setup_failed: STABLE_PEAK readback failed")


def verify_clocks(gpu_uuid: str | None = None) -> bool:
    """Return whether all selected AMD GPUs are in STABLE_PEAK mode."""
    locked = _all_gpus_at_level("STABLE_PEAK", gpu_uuid)
    if not locked:
        logger.warning("Not all selected GPUs are in STABLE_PEAK mode")
    return locked


def unlock_clocks(gpu_uuid: str | None = None) -> bool:
    """Reset GPU clocks to auto via ``amd-smi set -l AUTO``.

    Errors are logged but not raised. Returns whether every selected GPU was
    verified back in AUTO mode. None retains legacy all-device scope.
    """
    try:
        _run_checked_amd_smi(
            [
                "sudo",
                "-n",
                *clock_command(
                    _amd_smi_executable(gpu_uuid), ClockLevel.AUTO, gpu_uuid
                ),
            ],
        )
    except Exception as e:  # noqa: BLE001 -- Cleanup must never mask callers
        logger.warning("Failed to unlock clocks: %s", e)
        return False
    if not _all_gpus_at_level("AUTO", gpu_uuid):
        logger.warning("GPU clock reset could not be verified at AUTO")
        return False
    return True


def are_clocks_locked() -> bool:
    """Check whether clocks were locked successfully before evaluation."""
    return os.environ.get(ENV_SOL_EXECBENCH_CLOCKS_LOCKED, "0") == "1"
