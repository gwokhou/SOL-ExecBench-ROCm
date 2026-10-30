# SPDX-License-Identifier: Apache-2.0
"""Bounded command diagnostics for the existing clock lifecycle stderr log."""

import json
import logging
import os
import shutil
from enum import StrEnum
from pathlib import Path

from sol_execbench.core.platform.clock_commands import ClockLevel
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.timestamps import utc_timestamp

logger = logging.getLogger(__name__)

OUTPUT_LIMIT_CHARS = 4096
CONTEXT_LIMIT_CHARS = 1024
ENVIRONMENT_NAMES = (
    "PATH",
    "HOME",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "ROCM_PATH",
    "LD_LIBRARY_PATH",
)
PROCESS_STATUS_FIELDS = frozenset(
    {"Uid", "Gid", "Groups", "NoNewPrivs", "Seccomp"}
)


class ClockCommandPhase(StrEnum):
    """Command phases; AUTO includes both normal cleanup and rollback."""

    SETUP = "setup"
    RESTORE = "restore"


def _process_status() -> dict[str, str]:
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError as error:
        return {"read_error": type(error).__name__}
    return {
        name: redacted_text_tail(value.strip(), CONTEXT_LIMIT_CHARS)
        for line in lines
        for name, _, value in [line.partition(":")]
        if name in PROCESS_STATUS_FIELDS
    }


def _executable(name: str) -> dict[str, str | int | None]:
    located = shutil.which(name)
    if located is None:
        return {"invoked": name, "resolved": None}
    try:
        path = Path(located).resolve()
        metadata = path.stat()
        return {
            "invoked": name,
            "resolved": str(path),
            "inode": metadata.st_ino,
            "device": metadata.st_dev,
            "mode": metadata.st_mode & 0o7777,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
    except OSError as error:
        return {"invoked": name, "read_error": type(error).__name__}


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def log_clock_command(
    command: list[str],
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    *,
    error: str | None = None,
) -> None:
    """Emit one redacted JSON line; the foreground launcher persists stderr."""
    output, errors = _text(stdout), _text(stderr)
    event = {
        "at_utc": utc_timestamp(),
        "pid": os.getpid(),
        "phase": ClockCommandPhase.SETUP
        if command[5] == ClockLevel.PEAK
        else ClockCommandPhase.RESTORE,
        "command": command,
        "returncode": returncode if isinstance(returncode, int) else None,
        "error": error,
        "stdout": redacted_text_tail(output, OUTPUT_LIMIT_CHARS),
        "stderr": redacted_text_tail(errors, OUTPUT_LIMIT_CHARS),
        "stdout_truncated": len(output) > OUTPUT_LIMIT_CHARS,
        "stderr_truncated": len(errors) > OUTPUT_LIMIT_CHARS,
        "environment": {
            name: redacted_text_tail(os.environ[name], CONTEXT_LIMIT_CHARS)
            for name in ENVIRONMENT_NAMES
            if name in os.environ
        },
        "process_status": _process_status(),
        "executables": [_executable(command[0]), _executable(command[2])],
    }
    # WARNING is intentional: plain CLI execution must persist successes and
    # failures through the already attached stderr stream, without logger setup.
    logger.warning(
        "clock_command_diagnostic %s", json.dumps(event, sort_keys=True)
    )
