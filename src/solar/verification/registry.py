# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Executable verification backends for registered SOLAR IR dialects."""

from solar.ir.contracts import IRKind, normalize_ir_kind
from solar.ir.registry import ir_backend
from solar.verification.aten import execute_aten_layer
from solar.verification.contracts import (
    IRVerificationBackend,
    LayerExecutor,
)
from solar.verification.extended import execute_extended_einsum_layer

_EXECUTORS: dict[IRKind, LayerExecutor] = {
    IRKind.ATEN: execute_aten_layer,
    IRKind.EXTENDED_EINSUM: execute_extended_einsum_layer,
}


def verification_backend(kind: IRKind | str) -> IRVerificationBackend:
    """Return the verifier runtime paired with its exact IR backend."""
    normalized = normalize_ir_kind(kind)
    return IRVerificationBackend(
        ir=ir_backend(normalized),
        execute=_EXECUTORS[normalized],
    )


def verification_backends() -> tuple[IRVerificationBackend, ...]:
    """Return every registered executable verification backend."""
    return tuple(verification_backend(kind) for kind in IRKind)


__all__ = ["verification_backend", "verification_backends"]
