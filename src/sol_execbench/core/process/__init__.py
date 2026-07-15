# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Process-execution primitives shared across SOL ExecBench."""

from .logs import (
    DEFAULT_LOG_TAIL_CHARS,
    redacted_file_tail,
    redacted_text_tail,
    run_command_to_files,
    temporary_stream_path,
)
from .subprocesses import run_in_process_group
from .stdio import flush_stdio_streams

__all__ = [
    "DEFAULT_LOG_TAIL_CHARS",
    "flush_stdio_streams",
    "redacted_file_tail",
    "redacted_text_tail",
    "run_command_to_files",
    "run_in_process_group",
    "temporary_stream_path",
]
