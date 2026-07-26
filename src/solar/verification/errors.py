# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable failure types for strict SOLAR conversion verification."""


class VerificationError(ValueError):
    """The reference and einsum graph could not be proven equivalent."""


class EinsumExecutionError(VerificationError):
    """An einsum graph cannot be executed exactly by the built-in verifier."""


__all__ = ["EinsumExecutionError", "VerificationError"]
