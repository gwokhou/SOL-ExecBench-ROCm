#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Start the trusted reference service, then replace this process with the candidate."""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path

from sol_execbench.driver.reference_runtime_api import (
    REFERENCE_PID_ENV,
    REFERENCE_REQUEST_FD_ENV,
    REFERENCE_RESPONSE_FD_ENV,
    REFERENCE_TOKEN_ENV,
    TRUSTED_DEFINITION_FILE,
)

STAGING_DIR = Path(__file__).parent.resolve()
_token = secrets.token_hex(32)
_request_read, _request_write = os.pipe()
_response_read, _response_write = os.pipe()
_worker_environment = dict(os.environ)
_worker_environment[REFERENCE_REQUEST_FD_ENV] = str(_request_read)
_worker_environment[REFERENCE_RESPONSE_FD_ENV] = str(_response_write)
_worker_environment[REFERENCE_TOKEN_ENV] = _token

_worker = subprocess.Popen(
    [sys.executable, "reference_worker.py"],
    cwd=STAGING_DIR,
    env=_worker_environment,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    text=True,
    pass_fds=(_request_read, _response_write),
)
os.close(_request_read)
os.close(_response_write)
try:
    _ready = _worker.stdout.readline().strip() if _worker.stdout else ""
    if _ready != "READY":
        raise RuntimeError(
            f"trusted reference worker failed before readiness (exit={_worker.poll()})"
        )
    if _worker.stdout is not None:
        _worker.stdout.close()
    # The worker has loaded the trusted source into private process memory.
    # Remove the only staged copy before any candidate code can execute.
    (STAGING_DIR / TRUSTED_DEFINITION_FILE).unlink()
    os.set_inheritable(_request_write, True)
    os.set_inheritable(_response_read, True)
    _candidate_environment = dict(os.environ)
    _candidate_environment[REFERENCE_REQUEST_FD_ENV] = str(_request_write)
    _candidate_environment[REFERENCE_RESPONSE_FD_ENV] = str(_response_read)
    _candidate_environment[REFERENCE_TOKEN_ENV] = _token
    _candidate_environment[REFERENCE_PID_ENV] = str(_worker.pid)
    os.execve(
        sys.executable,
        [sys.executable, "eval_driver.py"],
        _candidate_environment,
    )
except BaseException:
    os.close(_request_write)
    os.close(_response_read)
    _worker.terminate()
    try:
        _worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _worker.kill()
        _worker.wait()
    raise
