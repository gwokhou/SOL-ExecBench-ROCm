# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Platform-specific, non-reaping process and process-group observation."""

# Runtime feature lookup is required because Linux and macOS expose disjoint APIs.
# ruff: noqa: B009

from __future__ import annotations

import ctypes
import errno
import os
import select
import subprocess
import sys
import time
from pathlib import Path

_WAIT_POLL_SECONDS = 0.01
_PROC_PIDTBSDINFO = 3
_DARWIN_ZOMBIE_STATUS = 5


class _DarwinBSDInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def wait_for_exit_without_reaping(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout: float | None,
) -> None:
    """Observe a child exit without releasing its PID/session identity."""
    if sys.platform == "darwin":
        _wait_with_kqueue(process, timeout)
        return
    _wait_with_waitid(process, timeout)


def _wait_with_waitid(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout: float | None,
) -> None:
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        raise RuntimeError(
            f"non-reaping process observation is unsupported on {sys.platform}",
        )
    flags = getattr(os, "WEXITED") | getattr(os, "WNOWAIT")
    if timeout is None:
        waitid(getattr(os, "P_PID"), process.pid, flags)
        return
    deadline = time.monotonic() + timeout
    flags |= getattr(os, "WNOHANG")
    while waitid(getattr(os, "P_PID"), process.pid, flags) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(_WAIT_POLL_SECONDS, remaining))


def _wait_with_kqueue(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout: float | None,
) -> None:
    kqueue_factory = getattr(select, "kqueue", None)
    kevent_factory = getattr(select, "kevent", None)
    if kqueue_factory is None or kevent_factory is None:
        raise RuntimeError("macOS kqueue process observation is unavailable")
    queue = kqueue_factory()
    try:
        event = kevent_factory(
            process.pid,
            filter=getattr(select, "KQ_FILTER_PROC"),
            flags=(
                getattr(select, "KQ_EV_ADD")
                | getattr(select, "KQ_EV_ENABLE")
                | getattr(select, "KQ_EV_ONESHOT")
            ),
            fflags=getattr(select, "KQ_NOTE_EXIT"),
        )
        try:
            events = queue.control([event], 1, timeout)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return
            raise
        if not events:
            raise subprocess.TimeoutExpired(process.args, timeout or 0)
    finally:
        queue.close()


def process_group_has_live_members(process_group_id: int) -> bool:
    """Return whether a session has a live member other than its leader."""
    if sys.platform.startswith("linux"):
        return _linux_process_group_has_live_members(process_group_id)
    if sys.platform == "darwin":
        return _darwin_process_group_has_live_members(process_group_id)
    raise RuntimeError(
        f"process-group observation is unsupported on {sys.platform}"
    )


def _linux_process_group_has_live_members(process_group_id: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == process_group_id:
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            state, process_group, session = (
                fields[0],
                int(fields[2]),
                int(fields[3]),
            )
        except (OSError, IndexError, ValueError):
            continue
        if (
            state != "Z"
            and process_group == process_group_id
            and session == process_group_id
        ):
            return True
    return False


def _darwin_process_group_has_live_members(process_group_id: int) -> bool:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    process_count = libproc.proc_listallpids(None, 0)
    if process_count <= 0:
        raise OSError(ctypes.get_errno(), "proc_listallpids failed")
    pids = (ctypes.c_int * (process_count + 32))()
    listed = libproc.proc_listallpids(pids, ctypes.sizeof(pids))
    if listed < 0:
        raise OSError(ctypes.get_errno(), "proc_listallpids failed")
    for pid in pids[:listed]:
        if pid <= 0 or pid == process_group_id:
            continue
        info = _DarwinBSDInfo()
        size = libproc.proc_pidinfo(
            pid,
            _PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if size != ctypes.sizeof(info):
            continue
        try:
            session = os.getsid(pid)
        except (PermissionError, ProcessLookupError):
            continue
        if (
            info.pbi_pgid == process_group_id
            and session == process_group_id
            and (info.pbi_status != _DARWIN_ZOMBIE_STATUS)
        ):
            return True
    return False
