# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable failure types for strict SOLAR conversion verification."""


class VerificationError(ValueError):
    """The reference and IR graph could not be proven equivalent."""


class IrExecutionError(VerificationError):
    """An IR graph cannot be executed exactly by the built-in verifier."""


__all__ = ["IrExecutionError", "VerificationError"]
