# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Required-evidence rules for SOLAR semantic groups."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.enums import OpFamily
from sol_execbench.core.scoring.solar_derivation.coverage import _unique_sorted
from sol_execbench.core.scoring.solar_derivation.evidence_models import (
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarFormulaEvidence,
    SolarTensorEvidence,
)


def _required_evidence_for_group(
    family: str,
    tensors: tuple[SolarTensorEvidence, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    formula_evidence: tuple[SolarFormulaEvidence, ...] = (),
    byte_evidence: tuple[SolarByteEvidence, ...] = (),
    bound_evidence: tuple[SolarBoundEvidence, ...] = (),
) -> tuple[str, ...]:
    required = []
    for tensor in tensors:
        if tensor.shape is not None:
            required.append(f"shape:{tensor.tensor_id}")
        if tensor.dtype and tensor.dtype != "unknown":
            required.append(f"dtype:{tensor.tensor_id}")
        if tensor.semantic_axes:
            required.append(f"semantic_axes:{tensor.tensor_id}")
        if tensor.source.kind and tensor.source.detail:
            required.append(f"source:{tensor.tensor_id}")
    for estimate in estimates:
        if estimate.formula_inputs:
            required.append(f"formula_inputs:{estimate.node_id}")
        if estimate.formula and estimate.formula != "0":
            required.append(f"formula:{estimate.node_id}")
        if estimate.total_bytes > 0.0:
            required.append(f"bytes:{estimate.node_id}")
        if estimate.axis_source is not None:
            required.append(f"axis:{estimate.node_id}")
    required.extend(
        f"formula_evidence:{evidence.node_id}" for evidence in formula_evidence
    )
    required.extend(f"byte_evidence:{evidence.node_id}" for evidence in byte_evidence)
    required.extend(f"bound_evidence:{evidence.node_id}" for evidence in bound_evidence)
    if family == OpFamily.MOE.value:
        for estimate in estimates:
            if estimate.formula_kind == "moe_static_route_flops":
                required.extend(
                    ("shape:tokens", "shape:hidden", "shape:experts", "route:top_k")
                )
            elif estimate.formula_kind == "moe_dynamic_route_bytes":
                if "tokens" in estimate.formula_inputs:
                    required.append("shape:tokens")
                if "hidden" in estimate.formula_inputs:
                    required.append("shape:hidden")
                if "experts" in estimate.formula_inputs:
                    required.append("shape:experts")
    if family == OpFamily.SSM_MAMBA.value:
        for estimate in estimates:
            if estimate.formula_kind == "ssm_mamba_static_scan_flops":
                required.extend(
                    ("shape:sequence", "shape:hidden", "shape:state", "subrole:scan")
                )
            elif estimate.formula_kind == "ssm_mamba_degraded_scan_bytes":
                if "sequence" in estimate.formula_inputs:
                    required.append("shape:sequence")
                if "hidden" in estimate.formula_inputs:
                    required.append("shape:hidden")
    return _unique_sorted(required)
