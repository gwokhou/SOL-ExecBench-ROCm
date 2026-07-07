# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Confidence classification for SOLAR derivation evidence."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraphNode,
    BoundTensor,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation_confidence_families import (
    _attention_confidence_evidence,
    _convolution_confidence_evidence,
    _embedding_positional_confidence_evidence,
    _moe_confidence_evidence,
    _ssm_mamba_confidence_evidence,
)
from sol_execbench.core.scoring.solar_derivation_confidence_tensors import (
    _tensor_dtype,
    _tensor_has_semantic_axes,
    _tensor_has_source,
    _tensor_id,
    _tensor_shape,
)
from sol_execbench.core.scoring.solar_derivation_coverage import (
    _status_for_confidence,
    _unique_sorted,
    _worse_confidence,
    _worst_estimate_confidence,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SolarConfidenceClassification,
    SolarTensorEvidence,
)

def classify_solar_confidence(
    *,
    family: OpFamily | str,
    nodes: tuple[BoundGraphNode, ...],
    tensors: tuple[BoundTensor | SolarTensorEvidence, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> SolarConfidenceClassification:
    """Classify semantic group evidence without external side effects."""
    family_value = family.value if isinstance(family, OpFamily) else str(family)
    missing: list[str] = []
    warning_prefixes: list[str] = []

    if family_value == OpFamily.UNSUPPORTED.value:
        missing.append("family:recognized")
    if not nodes:
        missing.append("node:visible")
    if not subrole_names:
        missing.append("subroles:core")
    if not tensors:
        missing.append("tensors:related")
    for tensor in sorted(tensors, key=_tensor_id):
        tensor_id = _tensor_id(tensor)
        if _tensor_shape(tensor) is None:
            missing.append(f"shape:{tensor_id}")
        if not _tensor_dtype(tensor) or _tensor_dtype(tensor) == "unknown":
            missing.append(f"dtype:{tensor_id}")
        if not _tensor_has_semantic_axes(tensor):
            missing.append(f"semantic_axes:{tensor_id}")
        if not _tensor_has_source(tensor):
            missing.append(f"source:{tensor_id}")

    if not estimates:
        missing.append("estimate:operator_work")
    for estimate in sorted(estimates, key=lambda item: item.node_id):
        if estimate.confidence == EstimateConfidence.UNSUPPORTED:
            missing.append(f"estimate:{estimate.node_id}")
        if not estimate.formula_inputs:
            missing.append(f"formula_inputs:{estimate.node_id}")
        if not estimate.formula or estimate.formula == "0":
            missing.append(f"formula:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"bytes:{estimate.node_id}")
        if estimate.axis_source is None:
            missing.append(f"axis:{estimate.node_id}")
        warning_prefixes.extend(estimate.warnings)

    if family_value == OpFamily.ATTENTION.value:
        attention_missing, attention_warnings = _attention_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(attention_missing)
        warning_prefixes.extend(attention_warnings)
    if family_value == OpFamily.CONVOLUTION.value:
        convolution_missing, convolution_warnings = _convolution_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(convolution_missing)
        warning_prefixes.extend(convolution_warnings)
    if family_value == OpFamily.EMBEDDING_POSITIONAL.value:
        memory_missing, memory_warnings = _embedding_positional_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(memory_missing)
        warning_prefixes.extend(memory_warnings)
    if family_value == OpFamily.MOE.value:
        moe_missing, moe_warnings = _moe_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(moe_missing)
        warning_prefixes.extend(moe_warnings)
    if family_value == OpFamily.SSM_MAMBA.value:
        ssm_missing, ssm_warnings = _ssm_mamba_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(ssm_missing)
        warning_prefixes.extend(ssm_warnings)

    confidence = _worst_estimate_confidence(estimates)
    if family_value == OpFamily.UNSUPPORTED.value or not nodes or not subrole_names:
        confidence = EstimateConfidence.UNSUPPORTED
    elif missing:
        confidence = _worse_confidence(confidence, EstimateConfidence.INEXACT)

    status = _status_for_confidence(confidence)
    if confidence == EstimateConfidence.INEXACT:
        warning_prefixes.append(f"inexact_operator:{family_value}")
        if family_value == OpFamily.ATTENTION.value:
            warning_prefixes.append("aggregate_degraded:attention")
        if family_value == OpFamily.MOE.value:
            warning_prefixes.append("aggregate_degraded:moe")
        if family_value == OpFamily.SSM_MAMBA.value:
            warning_prefixes.append("aggregate_degraded:ssm_mamba")
        warning_prefixes.append("aggregate_degraded:incomplete semantic evidence")
        rationale = (
            f"{family_value} semantics are visible but metadata is incomplete: "
            f"{', '.join(_unique_sorted(missing))}"
        )
    elif confidence == EstimateConfidence.UNSUPPORTED:
        warning_prefixes.append(f"unsupported_operator:{family_value}")
        if family_value == OpFamily.ATTENTION.value:
            warning_prefixes.append("aggregate_unscored:attention")
        if family_value == OpFamily.MOE.value:
            warning_prefixes.append("aggregate_unscored:moe")
        if family_value == OpFamily.SSM_MAMBA.value:
            warning_prefixes.append("aggregate_unscored:ssm_mamba")
        warning_prefixes.append("aggregate_unscored:unsupported semantic evidence")
        rationale = (
            f"{family_value} evidence is unsupported for scoring: "
            f"{', '.join(_unique_sorted(missing))}"
        )
    else:
        rationale = (
            f"{family_value} evidence has visible family, core subroles, tensor "
            "metadata, formula inputs, byte evidence, axis provenance, and source provenance"
        )

    return SolarConfidenceClassification(
        confidence=confidence,
        status=status,
        missing_evidence=_unique_sorted(missing),
        warning_prefixes=_unique_sorted(warning_prefixes),
        rationale=rationale,
    )
