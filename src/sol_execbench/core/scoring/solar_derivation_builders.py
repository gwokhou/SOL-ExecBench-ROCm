# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builders for SOLAR derivation evidence sidecars."""

from __future__ import annotations

from sol_execbench.core.data.definition import AxisConst, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
    build_bound_graph,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.solar_derivation_coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
    _default_source_boundary,
    _derivation_warnings,
    _status_for_confidence,
    _unique_sorted,
    _worse_confidence,
    _worst_estimate_confidence,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF,
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarConfidenceClassification,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)


def build_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
) -> SolarDerivationEvidence:
    """Build internal SOLAR derivation evidence from canonical problem inputs."""
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    return derive_solar_derivation_evidence(definition, workload, graph, estimates)


def derive_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
) -> SolarDerivationEvidence:
    """Derive SOLAR evidence from a prebuilt bound graph and operator estimates."""
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    tensors = tuple(
        _tensor_evidence(definition, workload, graph, tensor)
        for _, tensor in sorted(graph.tensors.items())
    )
    tensor_evidence_by_id = {tensor.tensor_id: tensor for tensor in tensors}
    groups = _semantic_group_evidence(
        graph,
        estimates,
        nodes_by_id=nodes_by_id,
        tensor_evidence_by_id=tensor_evidence_by_id,
    )
    warnings = _derivation_warnings(graph, estimates)
    return SolarDerivationEvidence(
        definition=definition.name,
        workload_uuid=workload.uuid,
        groups=groups,
        tensors=tensors,
        warnings=warnings,
        source_boundary=_default_source_boundary(),
        coverage_summary=_coverage_for_groups(groups),
        aggregate_status=_aggregate_status_for_groups(groups, warnings),
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


def _tensor_evidence(
    definition: Definition,
    workload: Workload,
    graph: BoundGraph,
    tensor: BoundTensor,
) -> SolarTensorEvidence:
    missing_evidence = []
    if tensor.shape is None:
        missing_evidence.append(f"shape:{tensor.tensor_id}")
    if not tensor.dtype or tensor.dtype == "unknown":
        missing_evidence.append(f"dtype:{tensor.tensor_id}")
    semantic_axes = _semantic_axes_for_tensor(definition, workload, tensor)
    if tensor.shape is not None and not semantic_axes:
        missing_evidence.append(f"semantic_axes:{tensor.tensor_id}")
    return SolarTensorEvidence(
        tensor_id=tensor.tensor_id,
        name=tensor.name,
        shape=tensor.shape,
        dtype=tensor.dtype,
        semantic_axes=semantic_axes,
        source=_source_for_tensor(graph, tensor),
        producer_node_id=tensor.producer_node_id,
        missing_evidence=tuple(missing_evidence),
    )


def _semantic_group_evidence(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    nodes_by_id: dict[str, BoundGraphNode],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSemanticGroupEvidence, ...]:
    estimates_by_family: dict[str, list[OperatorWorkEstimate]] = {}
    for estimate in estimates:
        estimates_by_family.setdefault(estimate.op_family.value, []).append(estimate)

    groups: list[SolarSemanticGroupEvidence] = []
    for group_index, (family, family_estimates) in enumerate(
        sorted(
            estimates_by_family.items(),
            key=lambda item: _first_estimate_node_id(item[1]),
        ),
        start=1,
    ):
        ordered_estimates = tuple(
            sorted(family_estimates, key=lambda item: item.node_id)
        )
        nodes = tuple(
            nodes_by_id[estimate.node_id]
            for estimate in ordered_estimates
            if estimate.node_id in nodes_by_id
        )
        node_ids = tuple(node.node_id for node in nodes)
        related_tensor_ids = _group_tensor_ids(nodes)
        related_tensors = tuple(
            tensor_evidence_by_id[tensor_id]
            for tensor_id in related_tensor_ids
            if tensor_id in tensor_evidence_by_id
        )
        subroles = _subroles_for_group(family, nodes, tensor_evidence_by_id)
        classification = classify_solar_confidence(
            family=family,
            nodes=nodes,
            tensors=related_tensors,
            estimates=ordered_estimates,
            subrole_names=tuple(subrole.name for subrole in subroles),
        )
        source = _source_for_group(family, ordered_estimates, nodes)
        formula_evidence = _formula_evidence_for_estimates(ordered_estimates)
        byte_evidence = _byte_evidence_for_estimates(
            ordered_estimates,
            nodes_by_id=nodes_by_id,
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        bound_evidence = _bound_evidence_for_estimates(ordered_estimates)
        groups.append(
            SolarSemanticGroupEvidence(
                family=family,
                group_id=f"group:{family}:{group_index}",
                node_ids=node_ids,
                subroles=subroles,
                confidence=classification.confidence,
                status=classification.status,
                required_evidence=_required_evidence_for_group(
                    family,
                    related_tensors,
                    ordered_estimates,
                    formula_evidence=formula_evidence,
                    byte_evidence=byte_evidence,
                    bound_evidence=bound_evidence,
                ),
                missing_evidence=classification.missing_evidence,
                warning_prefixes=classification.warning_prefixes,
                source=source,
                rationale=classification.rationale,
                formula_evidence=formula_evidence,
                byte_evidence=byte_evidence,
                bound_evidence=bound_evidence,
            )
        )
    return tuple(groups)


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


def _first_estimate_node_id(estimates: list[OperatorWorkEstimate]) -> str:
    return min(estimate.node_id for estimate in estimates)


def _group_tensor_ids(nodes: tuple[BoundGraphNode, ...]) -> tuple[str, ...]:
    tensor_ids: list[str] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        tensor_ids.extend(node.input_tensor_ids)
        tensor_ids.extend(node.output_tensor_ids)
    return tuple(dict.fromkeys(tensor_ids))


def _source_for_group(
    family: str,
    estimates: tuple[OperatorWorkEstimate, ...],
    nodes: tuple[BoundGraphNode, ...],
) -> SolarEvidenceSource:
    if estimates:
        first = estimates[0]
        return SolarEvidenceSource(
            kind="estimate",
            detail=f"{first.formula_kind}:{first.formula}",
            node_id=first.node_id,
            tensor_id=None,
        )
    if nodes:
        first_node = nodes[0]
        return SolarEvidenceSource(
            kind=_source_kind_for_node(first_node),
            detail=first_node.source_expression,
            node_id=first_node.node_id,
            tensor_id=None,
        )
    return SolarEvidenceSource(
        kind="ast",
        detail=f"unsupported group:{family}",
        node_id=None,
        tensor_id=None,
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


def _attention_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    qk_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "qk_scores"),
        None,
    )
    pv_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "pv_aggregation"),
        None,
    )
    if qk_node is not None and len(qk_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="q_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[0],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
        subroles.append(
            _subrole_from_tensor_ids(
                name="k_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    if pv_node is not None and len(pv_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="v_projection",
                node=pv_node,
                tensor_ids=(pv_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str) or subrole in {
            "dynamic_attention_axes",
        }:
            continue
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _attention_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    node_subroles = {
        str(node.attributes.get("subrole"))
        for node in nodes
        if node.attributes.get("subrole") is not None
    }
    if "dynamic_attention_axes" in node_subroles:
        missing.extend(("axis:static_sequence", "shape:sequence_q", "shape:sequence_k"))
        warnings.append("unsupported_operator:dynamic_attention_axes")
        return missing, warnings

    required = {
        "q_projection",
        "k_projection",
        "v_projection",
        "qk_scores",
        "softmax",
        "pv_aggregation",
    }
    missing.extend(f"attention_subrole:{name}" for name in sorted(required - subroles))
    if "output_projection" not in subroles:
        missing.append("attention_subrole:output_projection")

    softmax_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "softmax"
    ]
    if not softmax_nodes or all(
        node.attributes.get("axis") is None for node in softmax_nodes
    ):
        missing.append("axis:softmax")
    if any(node.attributes.get("mask_semantics") == "partial" for node in nodes):
        missing.extend(("mask:semantics", "mask:sparsity"))
        warnings.append("inexact_operator:attention_mask")

    attention_estimates = [
        estimate
        for estimate in estimates
        if estimate.op_family == OpFamily.ATTENTION
        and estimate.confidence != EstimateConfidence.UNSUPPORTED
    ]
    if not attention_estimates:
        missing.append("estimate:attention")
    for estimate in attention_estimates:
        if not estimate.formula_inputs:
            missing.append(f"attention_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"attention_bytes:{estimate.node_id}")
    return missing, warnings


def _convolution_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    required_subroles = {"input", "weight", "output", "convolution_metadata"}
    missing.extend(
        f"convolution_subrole:{name}" for name in sorted(required_subroles - subroles)
    )
    for node in nodes:
        for key in (
            "dimensionality",
            "stride",
            "padding",
            "dilation",
            "groups",
            "output_spatial",
        ):
            if key not in node.attributes:
                missing.append(f"convolution:{key}")
                warnings.append(f"inexact_operator:convolution_missing_{key}")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:convolution_missing_"):
                missing.append(
                    "convolution:"
                    + warning.removeprefix("inexact_operator:convolution_missing_")
                )
                warnings.append(warning)
        if (
            estimate.confidence != EstimateConfidence.UNSUPPORTED
            and not estimate.formula_inputs
        ):
            missing.append(f"convolution_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"convolution_bytes:{estimate.node_id}")
    return missing, warnings


def _embedding_positional_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if not subrole_names:
        missing.append("embedding_positional_subrole:memory_bound")
    for node in nodes:
        subrole = str(node.attributes.get("memory_subrole") or "")
        if subrole in {"embedding_lookup", "gather_lookup"}:
            for key in (
                "index_tensor_id",
                "index_dtype",
                "table_tensor_id",
                "table_shape",
                "output_shape",
                "selected_elements",
            ):
                if key not in node.attributes or node.attributes.get(key) is None:
                    missing.append(f"embedding_positional:{key}")
                    warnings.append(
                        f"inexact_operator:embedding_positional_missing_{key}"
                    )
        if subrole == "rotary_like" and len(node.input_tensor_ids) < 2:
            missing.append("embedding_positional:rotary_axes")
            warnings.append("inexact_operator:embedding_positional_missing_rotary_axes")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:embedding_positional_missing_"):
                missing.append(
                    "embedding_positional:"
                    + warning.removeprefix(
                        "inexact_operator:embedding_positional_missing_"
                    )
                )
                warnings.append(warning)
        if (
            estimate.confidence != EstimateConfidence.UNSUPPORTED
            and not estimate.formula_inputs
        ):
            missing.append(f"embedding_positional_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"embedding_positional_bytes:{estimate.node_id}")
    return missing, warnings


def _moe_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("taxonomy_only") for node in nodes):
        missing.extend(
            (
                "subrole:router",
                "subrole:expert_projection",
                "subrole:dispatch",
                "subrole:combine",
            )
        )
        warnings.append("unsupported_operator:moe_taxonomy_only")
        return missing, warnings

    subroles = set(subrole_names)
    required = {"router", "dispatch", "expert_projection", "combine"}
    missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    if "top_k" not in subroles:
        missing.append("route:top_k")
    dispatch_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "dispatch"
    ]
    if not dispatch_nodes:
        missing.append("route:static_cardinality")
    for node in dispatch_nodes:
        if not isinstance(node.attributes.get("token_count"), int):
            missing.append("shape:tokens")
        if not isinstance(node.attributes.get("hidden_size"), int):
            missing.append("shape:hidden")
        if not isinstance(node.attributes.get("expert_count"), int):
            missing.append("shape:experts")
        if not isinstance(node.attributes.get("route_top_k"), int):
            missing.append("route:top_k")
            missing.append("route:static_cardinality")
            warnings.append("inexact_operator:moe_dynamic_routing")
        for item in node.attributes.get("missing_route_metadata", ()):
            if isinstance(item, str):
                missing.append(item)
                if item == "route:top_k":
                    warnings.append("inexact_operator:moe_missing_top_k")
                elif item == "route:static_cardinality":
                    warnings.append("inexact_operator:moe_missing_static_cardinality")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:moe_") or warning.startswith(
                "unsupported_operator:moe_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "moe_static_route_flops":
            for key, evidence_name in (
                ("tokens", "shape:tokens"),
                ("hidden", "shape:hidden"),
                ("experts", "shape:experts"),
                ("top_k", "route:top_k"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
    return missing, warnings


def _ssm_mamba_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("custom_scan") for node in nodes):
        missing.extend(
            ("subrole:recognized_scan", "shape:state", "recurrence:update_formula")
        )
        warnings.append("unsupported_operator:ssm_custom_scan")
        return missing, warnings

    subroles = set(subrole_names)
    if "scan" not in subroles:
        missing.append("subrole:scan")
    if not any(node.attributes.get("recognized_scan") is True for node in nodes):
        missing.append("subrole:recognized_scan")

    has_state_update = "state_update" in subroles
    if not has_state_update:
        missing.extend(("shape:state", "recurrence:update_formula"))
        warnings.append("inexact_operator:ssm_missing_recurrence")

    if has_state_update:
        required = {
            "input_projection",
            "depthwise_convolution",
            "scan",
            "state_update",
            "gating",
            "output_projection",
        }
        missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    for node in nodes:
        subrole = node.attributes.get("subrole")
        if subrole in {"scan", "state_update"}:
            if not isinstance(node.attributes.get("sequence_length"), int):
                missing.append("shape:sequence")
            if not isinstance(node.attributes.get("hidden_size"), int):
                missing.append("shape:hidden")
        if subrole == "state_update":
            if "state_shape" not in node.attributes:
                missing.append("shape:state")
            if "state_update_parameters" not in node.attributes:
                missing.append("recurrence:update_formula")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:ssm_") or warning.startswith(
                "unsupported_operator:ssm_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "ssm_mamba_static_scan_flops":
            for key, evidence_name in (
                ("sequence", "shape:sequence"),
                ("hidden", "shape:hidden"),
                ("state", "shape:state"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
        elif estimate.formula_kind == "ssm_mamba_degraded_scan_bytes":
            missing.extend(("shape:state", "recurrence:update_formula"))
            warnings.append("inexact_operator:ssm_missing_recurrence")
    return missing, warnings


def _linear_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.input_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="input",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[0],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 1:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="weight_or_rhs",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[1],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 2:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="bias",
                    node=node,
                    tensor_ids=tuple(node.input_tensor_ids[2:]),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if node.output_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="output",
                    node=node,
                    tensor_ids=node.output_tensor_ids,
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _convolution_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.input_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="input",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[0],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 1:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="weight",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[1],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 2:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="bias",
                    node=node,
                    tensor_ids=tuple(node.input_tensor_ids[2:]),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if node.output_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="output",
                    node=node,
                    tensor_ids=node.output_tensor_ids,
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        subroles.append(
            _subrole_from_tensor_ids(
                name="convolution_metadata",
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _embedding_positional_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        name = str(
            node.attributes.get("memory_subrole") or node.op_name or "memory_bound"
        )
        subroles.append(
            _subrole_from_tensor_ids(
                name=name,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _moe_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.attributes.get("taxonomy_only"):
            continue
        names: list[str] = []
        subrole = node.attributes.get("subrole")
        if isinstance(subrole, str):
            names.append(subrole)
        for name in node.attributes.get("moe_subroles", ()):
            if isinstance(name, str):
                names.append(name)
        for name in dict.fromkeys(names):
            key = (name, node.node_id)
            if key in seen:
                continue
            seen.add(key)
            subroles.append(
                _subrole_from_tensor_ids(
                    name=name,
                    node=node,
                    tensor_ids=_node_tensor_ids(node),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    order = {
        "router": 0,
        "top_k": 1,
        "dispatch": 2,
        "expert_projection": 3,
        "combine": 4,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )


def _ssm_mamba_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    has_state_update = any(
        node.attributes.get("subrole") == "state_update" for node in nodes
    )
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str):
            continue
        if not has_state_update and subrole not in {
            "input_projection",
            "depthwise_convolution",
            "scan",
        }:
            continue
        key = (subrole, node.node_id)
        if key in seen:
            continue
        seen.add(key)
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    order = {
        "input_projection": 0,
        "depthwise_convolution": 1,
        "scan": 2,
        "state_update": 3,
        "gating": 4,
        "output_projection": 5,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )


def _op_name_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles = [
        _subrole_from_tensor_ids(
            name=node.op_name or node.op_family.value,
            node=node,
            tensor_ids=_node_tensor_ids(node),
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    return tuple(sorted(subroles, key=lambda item: item.name))


def _subrole_from_tensor_ids(
    *,
    name: str,
    node: BoundGraphNode,
    tensor_ids: tuple[str, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> SolarSubroleEvidence:
    missing = tuple(
        evidence
        for tensor_id in tensor_ids
        if tensor_id in tensor_evidence_by_id
        for evidence in tensor_evidence_by_id[tensor_id].missing_evidence
    )
    return SolarSubroleEvidence(
        name=name,
        node_ids=(node.node_id,),
        tensor_ids=tuple(tensor_ids),
        source=SolarEvidenceSource(
            kind=_source_kind_for_node(node),
            detail=node.source_expression,
            node_id=node.node_id,
            tensor_id=tensor_ids[0] if tensor_ids else None,
        ),
        confidence=node.confidence,
        rationale=node.rationale,
        missing_evidence=_unique_sorted(missing),
    )


def _source_for_tensor(graph: BoundGraph, tensor: BoundTensor) -> SolarEvidenceSource:
    producer = (
        next(
            (node for node in graph.nodes if node.node_id == tensor.producer_node_id),
            None,
        )
        if tensor.producer_node_id is not None
        else None
    )
    kind = _source_kind_for_tensor(tensor, producer)
    return SolarEvidenceSource(
        kind=kind,
        detail=tensor.source,
        node_id=tensor.producer_node_id,
        tensor_id=tensor.tensor_id,
    )


def _source_kind_for_tensor(
    tensor: BoundTensor,
    producer: BoundGraphNode | None,
) -> str:
    if tensor.source.startswith("definition."):
        return "definition"
    if tensor.source.startswith("workload."):
        return "workload"
    if producer is not None and producer.attributes.get("trace_source") == "torch.fx":
        return "fx"
    if tensor.source.startswith("tmp:") and producer is not None:
        return _source_kind_for_node(producer)
    return "ast"


def _source_kind_for_node(node: BoundGraphNode) -> str:
    if node.attributes.get("trace_source") == "torch.fx":
        return "fx"
    return "ast"


def _semantic_axes_for_tensor(
    definition: Definition,
    workload: Workload,
    tensor: BoundTensor,
) -> tuple[str, ...]:
    spec = definition.inputs.get(tensor.name) or definition.outputs.get(tensor.name)
    if spec is not None and spec.shape is not None:
        return tuple(str(axis) for axis in spec.shape)
    if tensor.shape is None:
        return ()
    matched_axes = _axes_matching_shape(definition, workload, tensor.shape)
    if len(matched_axes) == len(tensor.shape):
        return matched_axes
    return ()


def _axes_matching_shape(
    definition: Definition,
    workload: Workload,
    shape: tuple[int, ...],
) -> tuple[str, ...]:
    axis_values = _axis_values(definition, workload)
    axes: list[str] = []
    for dim in shape:
        matching = [name for name, value in axis_values.items() if value == dim]
        if not matching:
            return ()
        axes.append(matching[0])
    return tuple(axes)


def _axis_values(definition: Definition, workload: Workload) -> dict[str, int]:
    values = {name: int(value) for name, value in workload.axes.items()}
    for name, axis in definition.axes.items():
        if isinstance(axis, AxisConst):
            values[name] = int(axis.value)
    return values


def _node_tensor_ids(node: BoundGraphNode | None) -> tuple[str, ...]:
    if node is None:
        return ()
    return tuple(dict.fromkeys((*node.input_tensor_ids, *node.output_tensor_ids)))


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
