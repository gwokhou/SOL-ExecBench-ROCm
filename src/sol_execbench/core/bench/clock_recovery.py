# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Clear a recovery gate only after live identity, idleness and AUTO checks.

This command never changes GPU policy or kills processes. An operator must first
stop the abandoned workload and restore the recorded devices using AMD-SMI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.bench.clock_ownership import (
    ClockOwnerRecord,
    ClockOwnershipError,
    ClockOwnerState,
    clock_owner_path,
    observe_clock_devices,
)
from sol_execbench.core.bench.gpu_lock import (
    GPU_LOCK_FILENAME,
    gpu_lock_directory,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.platform.amd_smi import (
    AMDSMIProcessPayload,
    parse_performance_levels,
    parse_processes,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process.locks import exclusive_file_lock
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.timestamps import utc_timestamp
from sol_execbench.tools.docker_process import (
    CONTAINER_CLEANUP_BLOCKER_PREFIX,
    CONTAINER_ID_SUFFIX,
    confirm_container_absent,
)

RECOVERY_TIMEOUT_SECONDS = 10
RECOVERY_OUTPUT_BYTES = 16384


def _read_device(command: str, bdf: str, *arguments: str) -> str:
    result = run_in_process_group_bounded(
        [
            resolve_rocm_tool_command("amd-smi"),
            command,
            "--gpu",
            bdf,
            *arguments,
            "--json",
        ],
        timeout=RECOVERY_TIMEOUT_SECONDS,
        max_capture_bytes=RECOVERY_OUTPUT_BYTES,
    )
    if result.returncode:
        raise ClockOwnershipError("clock_recovery_readback_failed")
    return result.stdout


def confirm_clock_recovery(directory: Path, token: str) -> None:
    """Confirm recovery under GPU/owner exclusion, preserving the old receipt."""
    path = clock_owner_path(directory, token)
    with (
        exclusive_file_lock(
            directory / GPU_LOCK_FILENAME.format(device_index=0)
        ),
        exclusive_file_lock(path.with_suffix(".lock")),
    ):
        record = ClockOwnerRecord.model_validate_json(path.read_text())
        if record.lease_id != token:
            raise ClockOwnershipError("clock_recovery_identity_mismatch")
        if record.container_supervised:
            confirm_container_absent(
                "docker",
                path.with_suffix(CONTAINER_ID_SUFFIX).read_text().strip(),
            )
        if record.release_blocker is not None:
            if not record.release_blocker.startswith(
                CONTAINER_CLEANUP_BLOCKER_PREFIX
            ):
                raise ClockOwnershipError("clock_recovery_blocker_unresolved")
            confirm_container_absent(
                "docker",
                record.release_blocker.removeprefix(
                    CONTAINER_CLEANUP_BLOCKER_PREFIX
                ),
            )
        if not set(record.devices).issubset(observe_clock_devices()):
            raise ClockOwnershipError("clock_recovery_device_mismatch")
        observations = []
        for device in record.devices:
            processes = _read_device("process", device.bdf)
            observations.append(processes)
            if len(
                AMDSMIProcessPayload.model_validate_json(processes).root
            ) != 1 or parse_processes(processes):
                raise ClockOwnershipError("clock_recovery_device_busy")
            metric = _read_device("metric", device.bdf, "--perf-level")
            observations.append(metric)
            levels = parse_performance_levels(metric)
            if levels != ("AMDSMI_DEV_PERF_LEVEL_AUTO",):
                raise ClockOwnershipError("clock_recovery_auto_unverified")
        recovered = record.model_copy(
            update={
                "state": ClockOwnerState.RESTORED,
                "recovered_at": utc_timestamp(),
                "recovery_observations": tuple(observations),
            }
        )
        # Keep the failed/crashed receipt as evidence; never delete the gate blindly.
        original = path.parent / "recovery-history" / path.name
        if not original.exists():
            atomic_write_json_value(original, record.model_dump(mode="json"))
        atomic_write_json_value(path, recovered.model_dump(mode="json"))


def main() -> None:
    """Read-only hardware verification of a manually restored owner lease."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lease_id")
    args = parser.parse_args()
    confirm_clock_recovery(gpu_lock_directory(), args.lease_id)


if __name__ == "__main__":
    main()
