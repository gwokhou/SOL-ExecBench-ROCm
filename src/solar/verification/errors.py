# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable failure types for strict SOLAR conversion verification."""

from solar.errors import ConversionVerificationError, IRReplayError

# Kept as the public compatibility name for the typed verification base.
VerificationError = ConversionVerificationError


class IRExecutionError(IRReplayError):
    """An IR graph cannot be executed exactly by the built-in verifier."""


__all__ = ["IRExecutionError", "VerificationError"]
