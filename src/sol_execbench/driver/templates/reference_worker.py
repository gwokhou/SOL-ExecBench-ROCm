#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted reference subprocess; this process never imports candidate code."""

from __future__ import annotations

import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import torch

from sol_execbench.core.platform.runtime import pin_cuda_device
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_INPUT_NONCE,
)
from sol_execbench.driver.reference_runtime_api import (
    REFERENCE_REQUEST_FD_ENV,
    REFERENCE_RESPONSE_FD_ENV,
    REFERENCE_TOKEN_ENV,
)
from sol_execbench.driver.reference_worker_api import serve_reference_worker

STAGING_DIR = Path(__file__).parent.resolve()
_device = os.environ.get(
    "SOL_EXECBENCH_DEVICE",
    "cuda:0" if torch.cuda.is_available() else "cpu",
)
# Pin the active device so reference outputs are always produced on the same
# device the candidate is timed on (device-b3 parity with the eval driver).
pin_cuda_device(_device)
_token = os.environ[REFERENCE_TOKEN_ENV]
_input_nonce = os.environ.pop(ENV_SOL_EXECBENCH_INPUT_NONCE)
_request_stream = Connection(
    int(os.environ[REFERENCE_REQUEST_FD_ENV]),
    readable=True,
    writable=False,
)
_response_stream = Connection(
    int(os.environ[REFERENCE_RESPONSE_FD_ENV]),
    readable=False,
    writable=True,
)

try:
    serve_reference_worker(
        STAGING_DIR,
        request_stream=_request_stream,
        response_stream=_response_stream,
        token=_token,
        input_nonce=_input_nonce,
        device=_device,
        ready_stream=sys.stdout,
    )
finally:
    _request_stream.close()
    _response_stream.close()
