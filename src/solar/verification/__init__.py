# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Numerical verification for the source-to-SOL trust chain."""

from typing import TYPE_CHECKING

from solar.verification.contracts import TolerancePolicy, VerificationPolicy

if TYPE_CHECKING:
    from solar.verification.verify import (
        IRExecutionError,
        IRGraphExecutor,
        VerificationError,
        create_verification_artifact,
        replay_verification_artifact,
        verify_callable_conversion,
    )

_LAZY_IMPORTS = {
    name: ("solar.verification.verify", name)
    for name in (
        "IRExecutionError",
        "IRGraphExecutor",
        "VerificationError",
        "create_verification_artifact",
        "replay_verification_artifact",
        "verify_callable_conversion",
    )
}


def __getattr__(name: str) -> object:
    """Load verification execution helpers without creating import cycles."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "IRExecutionError",
    "IRGraphExecutor",
    "TolerancePolicy",
    "VerificationError",
    "VerificationPolicy",
    "create_verification_artifact",
    "replay_verification_artifact",
    "verify_callable_conversion",
]
