# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Registered lifecycle backend for the NVLabs einsum IR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solar.common.types import DynamicValue
from solar.graph.contracts import ExtractionKind
from solar.ir.contracts import IRBackend, IRKind
from solar.ir.nvlabs_einsum.conversion import (
    convert_operator_graph,
    validate_nvlabs_einsum_graph,
)


def _execute(
    layer_id: str,
    layer: Mapping[str, DynamicValue],
    operands: Sequence[DynamicValue],
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    from solar.verification.nvlabs_einsum import execute_layer

    return execute_layer(
        layer_id,
        layer,
        operands,
        output_shapes=output_shapes,
    )


backend = IRBackend(
    kind=IRKind.NVLABS_EINSUM,
    extractions=frozenset(ExtractionKind),
    validate=validate_nvlabs_einsum_graph,
    convert=convert_operator_graph,
    execute=_execute,
)

__all__ = ["backend"]
