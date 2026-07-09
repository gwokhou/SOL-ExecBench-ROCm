# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tensor predicate helpers for SOLAR confidence classification."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundTensor,
    BoundTensorRole,
)
from sol_execbench.core.scoring.solar_derivation_evidence_models import SolarTensorEvidence

def _tensor_id(tensor: BoundTensor | SolarTensorEvidence) -> str:
    return tensor.tensor_id


def _tensor_shape(tensor: BoundTensor | SolarTensorEvidence) -> tuple[int, ...] | None:
    return tensor.shape


def _tensor_dtype(tensor: BoundTensor | SolarTensorEvidence) -> str:
    return tensor.dtype


def _tensor_has_source(tensor: BoundTensor | SolarTensorEvidence) -> bool:
    if isinstance(tensor, SolarTensorEvidence):
        return bool(tensor.source.kind and tensor.source.detail)
    return bool(tensor.source)


def _tensor_has_semantic_axes(tensor: BoundTensor | SolarTensorEvidence) -> bool:
    if tensor.shape is None:
        return False
    if isinstance(tensor, SolarTensorEvidence):
        return bool(tensor.semantic_axes)
    if tensor.role in {BoundTensorRole.INPUT, BoundTensorRole.OUTPUT}:
        return tensor.source.startswith(("definition.", "workload."))
    return False
