# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Minimal ROCTx control boundary for diagnostic counter replay."""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

_ROCTX_LIBRARY_CANDIDATES = (
    "librocprofiler-sdk-roctx.so",
    "libroctx64.so",
    "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
    "/opt/rocm/lib/libroctx64.so",
)


class ROCTxUnavailableError(RuntimeError):
    """ROCTx could not satisfy the required replay control contract."""


def _load_roctx() -> ctypes.CDLL:
    discovered = ctypes.util.find_library("rocprofiler-sdk-roctx")
    candidates = (
        (discovered,) if discovered else ()
    ) + _ROCTX_LIBRARY_CANDIDATES
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    raise ROCTxUnavailableError("ROCTx control library is unavailable")


class ROCTxReplayController:
    """Global pause/resume plus thread-local marker ranges."""

    def __init__(self, library: ctypes.CDLL | None = None) -> None:
        """Load ROCTx and bind the narrow API used by replay."""
        self._library = library or _load_roctx()
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._library.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        self._library.roctxProfilerPause.restype = ctypes.c_int
        self._library.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        self._library.roctxProfilerResume.restype = ctypes.c_int
        self._library.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self._library.roctxRangePushA.restype = ctypes.c_int
        self._library.roctxRangePop.argtypes = []
        self._library.roctxRangePop.restype = ctypes.c_int

    def pause(self) -> None:
        """Pause all profiler contexts in this process."""
        if self._library.roctxProfilerPause(0) != 0:
            raise ROCTxUnavailableError("ROCTx profiler pause failed")

    def run_range(
        self,
        name: str,
        fn: Callable[_P, _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        """Resume, execute one synchronized range, and pause fail-closed."""
        if self._library.roctxProfilerResume(0) != 0:
            raise ROCTxUnavailableError("ROCTx profiler resume failed")
        level = self._library.roctxRangePushA(name.encode())
        if level < 0:
            self.pause()
            raise ROCTxUnavailableError("ROCTx range push failed")
        try:
            result = fn(*args, **kwargs)
            _synchronize_gpu()
            return result
        finally:
            popped = self._library.roctxRangePop()
            self.pause()
            if popped < 0:
                raise ROCTxUnavailableError("ROCTx range pop failed")


def _synchronize_gpu() -> None:
    """Synchronize through the ROCm-compatible PyTorch device API."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def roctx_library_available() -> bool:
    """Return whether the required ROCTx control API can be loaded."""
    try:
        ROCTxReplayController()
    except (AttributeError, OSError, ROCTxUnavailableError):
        return False
    return True


__all__ = [
    "ROCTxReplayController",
    "ROCTxUnavailableError",
    "roctx_library_available",
]
