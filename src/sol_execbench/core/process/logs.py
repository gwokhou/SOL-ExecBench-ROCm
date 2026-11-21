# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Safe subprocess stream capture and credential-redacted log tails."""

from __future__ import annotations

import io
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.process.stdio import flush_stdio_streams

DEFAULT_LOG_TAIL_CHARS = 4000
MAX_EMBEDDED_LOG_BYTES = 64 * 1024

_TOKEN_PATTERN = re.compile(
    r"(?x)"
    r"(?:"
    r"(?i:"
    r"("
    r"[\"\']?"
    r"(?:(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL)[A-Z0-9_-]{0,64}?"
    r"|(?<![A-Z0-9_-])authorization)"
    r")"
    r"([\"\']?\s*[:=]\s*[\"\']?(?:bearer\s+)?)"
    r"([^\s\"\',}]+)"
    r")"
    r"|"
    r"(?<![A-Za-z0-9_-])"
    r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|hf_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]{16,})"
    r")",
)
_TOKEN_PREFIX_OVERLAP_CHARS = 512
_TOKEN_VALUE_DELIMITERS = frozenset(" \t\r\n'\",}")


def _replace_token_match(match: re.Match[str]) -> str:
    if match.group(1) is not None:
        return f"{match.group(1)}{match.group(2)}<redacted>"
    return "<redacted>"


def redacted_text_tail(value: str, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    """Return the bounded tail of *value* with common credentials redacted."""
    if limit <= 0:
        return ""
    window = value[-(limit + _TOKEN_PREFIX_OVERLAP_CHARS) :]
    redacted = _TOKEN_PATTERN.sub(_replace_token_match, window)
    return redacted[-limit:]


def _append_tail(tail: str, text: str, limit: int) -> str:
    if not text:
        return tail
    return (tail + text)[-limit:]


def _seek_near_tail(handle: io.BufferedReader, limit: int) -> None:
    info = os.fstat(handle.fileno())
    seek_span = limit * 4 + _TOKEN_PREFIX_OVERLAP_CHARS * 2
    if not stat.S_ISREG(info.st_mode) or info.st_size <= seek_span:
        return
    handle.seek(info.st_size - seek_span)
    prefix = handle.peek(4)[:4]
    skip = 0
    while skip < len(prefix) and 0x80 <= prefix[skip] < 0xC0:
        skip += 1
    if skip:
        handle.read(skip)


def _redacted_file_tail(path: Path, limit: int) -> str:
    tail = ""
    pending = ""
    in_secret = False
    chunk_size = max(limit, 8192)

    with path.open("rb") as handle:
        _seek_near_tail(handle, limit)
        for raw_chunk in iter(lambda: handle.read(chunk_size), b""):
            text = raw_chunk.decode(errors="replace")
            if in_secret:
                delimiter_index = next(
                    (
                        index
                        for index, char in enumerate(text)
                        if char in _TOKEN_VALUE_DELIMITERS
                    ),
                    None,
                )
                if delimiter_index is None:
                    continue
                tail = _append_tail(tail, text[delimiter_index], limit)
                text = text[delimiter_index + 1 :]
                in_secret = False

            pending += text
            while pending:
                match = _TOKEN_PATTERN.search(pending)
                if match is None:
                    if len(pending) > _TOKEN_PREFIX_OVERLAP_CHARS:
                        emit = pending[:-_TOKEN_PREFIX_OVERLAP_CHARS]
                        tail = _append_tail(tail, emit, limit)
                        pending = pending[-_TOKEN_PREFIX_OVERLAP_CHARS:]
                    break

                tail = _append_tail(tail, pending[: match.start()], limit)
                tail = _append_tail(tail, _replace_token_match(match), limit)
                if match.end() == len(pending):
                    pending = ""
                    in_secret = True
                    break
                pending = pending[match.end() :]

    if not in_secret:
        tail = _append_tail(
            tail,
            redacted_text_tail(pending, max(limit, 8192)),
            limit,
        )
    return tail


def redacted_file_tail(path: Path, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    """Return a bounded, credential-redacted tail from *path*."""
    if limit <= 0:
        return ""
    try:
        return _redacted_file_tail(path, limit)
    except OSError:
        return ""


def read_bounded_log(
    log_path: str | None,
    *,
    limit: int = MAX_EMBEDDED_LOG_BYTES,
) -> str | None:
    """Read a bounded UTF-8 log after flushing inherited stdio streams."""
    if not log_path:
        return None
    flush_stdio_streams()
    try:
        with Path(log_path).open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        return None
    truncated = len(data) > limit
    text = data[:limit].decode("utf-8", errors="replace")
    return f"{text}\n\n[log truncated]\n" if truncated else text


def temporary_stream_path(
    temp_dir: Path,
    name: str,
    stream_name: str,
    *,
    name_prefix: str = "",
) -> Path:
    """Allocate a persistent temporary log path with a filesystem-safe prefix."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    with tempfile.NamedTemporaryFile(
        prefix=f"{name_prefix}{safe_name}_{stream_name}_",
        suffix=".log",
        dir=temp_dir,
        delete=False,
    ) as handle:
        return Path(handle.name)


def run_command_to_files(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run *command* with stdout and stderr redirected to separate files."""
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = runner(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )

    if completed.stdout:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


__all__ = [
    "DEFAULT_LOG_TAIL_CHARS",
    "MAX_EMBEDDED_LOG_BYTES",
    "read_bounded_log",
    "redacted_file_tail",
    "redacted_text_tail",
    "run_command_to_files",
    "temporary_stream_path",
]
