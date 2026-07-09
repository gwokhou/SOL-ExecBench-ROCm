# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operator-derived evidence builders for SOLAR semantic groups."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraphNode
from sol_execbench.core.scoring.amd_hardware_models import default_amd_hardware_models
from sol_execbench.core.scoring.solar_derivation.evidence_models import (
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation.models import (
    SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF,
)
from sol_execbench.core.scoring.solar_derivation.sources import _node_tensor_ids


def _formula_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[SolarFormulaEvidence, ...]:
    evidence = [
        SolarFormulaEvidence(
            node_id=estimate.node_id,
            family=estimate.op_family.value,
            formula_kind=estimate.formula_kind,
            formula=estimate.formula,
            formula_inputs=dict(estimate.formula_inputs),
            source=SolarEvidenceSource(
                kind="estimate",
                detail=f"{estimate.formula_kind}:{estimate.formula}",
                node_id=estimate.node_id,
                tensor_id=None,
            ),
            confidence=estimate.confidence,
            rationale=estimate.rationale,
        )
        for estimate in estimates
        if estimate.formula and estimate.formula != "0"
    ]
    return tuple(sorted(evidence, key=lambda item: item.node_id))


def _byte_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    nodes_by_id: dict[str, BoundGraphNode],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarByteEvidence, ...]:
    evidence: list[SolarByteEvidence] = []
    for estimate in estimates:
        if estimate.total_bytes <= 0.0:
            continue
        node = nodes_by_id.get(estimate.node_id)
        tensor_ids = _node_tensor_ids(node)
        dtype_inputs = {
            tensor_id: tensor_evidence_by_id[tensor_id].dtype
            for tensor_id in tensor_ids
            if tensor_id in tensor_evidence_by_id
        }
        evidence.append(
            SolarByteEvidence(
                node_id=estimate.node_id,
                family=estimate.op_family.value,
                read_bytes=estimate.read_bytes,
                write_bytes=estimate.write_bytes,
                intermediate_bytes=estimate.intermediate_bytes,
                movement_bytes=estimate.movement_bytes,
                total_bytes=estimate.total_bytes,
                dtype_inputs=dtype_inputs,
                tensor_ids=tensor_ids,
                source=SolarEvidenceSource(
                    kind="estimate",
                    detail=f"{estimate.movement_kind or 'bytes'}:{estimate.total_bytes}",
                    node_id=estimate.node_id,
                    tensor_id=None,
                ),
                confidence=estimate.confidence,
                rationale=estimate.rationale,
            )
        )
    return tuple(sorted(evidence, key=lambda item: item.node_id))


def _bound_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[SolarBoundEvidence, ...]:
    hardware_model = default_amd_hardware_models()["gfx1200"]
    evidence: list[SolarBoundEvidence] = []
    for estimate in estimates:
        compute_bound_ms = (
            estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
            if hardware_model.peak_tflops > 0.0
            else 0.0
        )
        memory_bound_ms = (
            estimate.total_bytes
            / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0)
            * 1000.0
            if hardware_model.memory_bandwidth_gbps > 0.0
            else 0.0
        )
        if (
            hardware_model.peak_tflops <= 0.0
            and hardware_model.memory_bandwidth_gbps <= 0.0
        ):
            limiting_resource = "none"
        else:
            limiting_resource = (
                "compute" if compute_bound_ms >= memory_bound_ms else "memory"
            )
        evidence.append(
            SolarBoundEvidence(
                node_id=estimate.node_id,
                family=estimate.op_family.value,
                compute_bound_ms=compute_bound_ms,
                memory_bound_ms=memory_bound_ms,
                limiting_resource=limiting_resource,
                sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
                source=SolarEvidenceSource(
                    kind="estimate",
                    detail=f"amd_sol_v2:{SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF}",
                    node_id=estimate.node_id,
                    tensor_id=None,
                ),
                confidence=estimate.confidence,
                rationale=estimate.rationale,
            )
        )
    return tuple(sorted(evidence, key=lambda item: item.node_id))
