# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable execution and verification boundary for NVLabs einsum IR."""

from solar._vendor.nvlabs.verification.nvlabs_einsum import (
    EinsumExecutionError,
    EinsumGraphExecutor,
    VerificationError,
    create_verification_artifact,
    execute_layer,
    replay_verification_artifact,
    verify_callable_conversion,
    verify_generated_handler,
)

__all__ = [
    "EinsumExecutionError",
    "EinsumGraphExecutor",
    "VerificationError",
    "create_verification_artifact",
    "execute_layer",
    "replay_verification_artifact",
    "verify_callable_conversion",
    "verify_generated_handler",
]
