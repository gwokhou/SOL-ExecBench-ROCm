# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Logging isolation for Torch semantic graph providers."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections.abc import Iterator
import logging
from logging.handlers import QueueHandler
from queue import Queue
from typing import cast


_TORCH_EXPORT_LOGGER = logging.getLogger(f"{__name__}.torch_export")
_TORCH_EXPORT_LOGGER.setLevel(logging.INFO)
_TORCH_EXPORT_LOGGER.addHandler(logging.NullHandler())
_TORCH_EXPORT_LOGGER.propagate = False


class _LoggerTextStream:
    """Turn complete stdout/stderr lines into asynchronous logging records."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._pending = ""

    def write(self, message: str) -> int:
        self._pending += message
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if line:
                self._logger.log(self._level, "%s", line)
        return len(message)

    def flush(self) -> None:
        if self._pending:
            self._logger.log(self._level, "%s", self._pending)
            self._pending = ""


def configure_torch_export_diagnostics(
    log_target: object | None,
) -> tuple[list[logging.Handler], int, bool]:
    """Route export diagnostics to a queue or a caller-supplied handler."""
    previous = (
        list(_TORCH_EXPORT_LOGGER.handlers),
        _TORCH_EXPORT_LOGGER.level,
        _TORCH_EXPORT_LOGGER.propagate,
    )
    _TORCH_EXPORT_LOGGER.handlers.clear()
    handler: logging.Handler
    if isinstance(log_target, logging.Handler):
        handler = log_target
    elif log_target is not None:
        handler = QueueHandler(cast(Queue[logging.LogRecord], log_target))
    else:
        handler = logging.NullHandler()
    _TORCH_EXPORT_LOGGER.addHandler(handler)
    _TORCH_EXPORT_LOGGER.setLevel(logging.INFO)
    _TORCH_EXPORT_LOGGER.propagate = False
    return previous


def restore_torch_export_diagnostics(
    state: tuple[list[logging.Handler], int, bool],
) -> None:
    """Restore a logger state returned by configure_torch_export_diagnostics."""
    handlers, level, propagate = state
    _TORCH_EXPORT_LOGGER.handlers.clear()
    _TORCH_EXPORT_LOGGER.handlers.extend(handlers)
    _TORCH_EXPORT_LOGGER.setLevel(level)
    _TORCH_EXPORT_LOGGER.propagate = propagate


@contextmanager
def redirect_torch_export_output() -> Iterator[None]:
    """Keep third-party export diagnostics out of stdout without dropping them."""
    stdout = _LoggerTextStream(_TORCH_EXPORT_LOGGER, logging.INFO)
    stderr = _LoggerTextStream(_TORCH_EXPORT_LOGGER, logging.WARNING)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()
