# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Semantic subrole inference for SOLAR derivation evidence."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraphNode, OpFamily
from sol_execbench.core.scoring.solar_derivation.evidence_models import (
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation.subrole_attention import (
    _attention_subroles,
)
from sol_execbench.core.scoring.solar_derivation.subrole_complex import (
    _embedding_positional_subroles,
    _moe_subroles,
    _ssm_mamba_subroles,
)
from sol_execbench.core.scoring.solar_derivation.subrole_convolution import (
    _convolution_subroles,
)
from sol_execbench.core.scoring.solar_derivation.subrole_linear import (
    _linear_subroles,
    _op_name_subroles,
)


def _subroles_for_group(
    family: str,
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    if family == OpFamily.ATTENTION.value:
        return _attention_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.CONVOLUTION.value:
        return _convolution_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.EMBEDDING_POSITIONAL.value:
        return _embedding_positional_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.MOE.value:
        return _moe_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.SSM_MAMBA.value:
        return _ssm_mamba_subroles(nodes, tensor_evidence_by_id)
    if family in {OpFamily.GEMM.value, OpFamily.LINEAR_PROJECTION.value}:
        return _linear_subroles(nodes, tensor_evidence_by_id)
    if family in {
        OpFamily.SOFTMAX.value,
        OpFamily.DATA_MOVEMENT.value,
        OpFamily.DTYPE_CONVERSION.value,
        OpFamily.REDUCTION.value,
        OpFamily.ELEMENTWISE.value,
        OpFamily.MLP_ACTIVATION.value,
        OpFamily.NORMALIZATION.value,
    }:
        return _op_name_subroles(nodes, tensor_evidence_by_id)
    return ()
