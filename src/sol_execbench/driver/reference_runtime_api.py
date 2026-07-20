# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable import surface for generated trusted-reference processes."""

from sol_execbench.core.bench.reference_protocol import (
    REFERENCE_PID_ENV,
    REFERENCE_REQUEST_FD_ENV,
    REFERENCE_RESPONSE_FD_ENV,
    REFERENCE_TOKEN_ENV,
    TRUSTED_DEFINITION_FILE,
    ReferenceCase,
    ReferenceClient,
    ReferenceExecutionError,
    ReferenceFailureKind,
    ReferenceProtocolError,
    ReferenceTimingCase,
    connect_reference_worker,
)

__all__ = [
    "REFERENCE_PID_ENV",
    "REFERENCE_REQUEST_FD_ENV",
    "REFERENCE_RESPONSE_FD_ENV",
    "REFERENCE_TOKEN_ENV",
    "TRUSTED_DEFINITION_FILE",
    "ReferenceCase",
    "ReferenceClient",
    "ReferenceExecutionError",
    "ReferenceFailureKind",
    "ReferenceProtocolError",
    "ReferenceTimingCase",
    "connect_reference_worker",
]
