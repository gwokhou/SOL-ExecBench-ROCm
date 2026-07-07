# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AMD SOL bound artifact v2 sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    EstimateConfidence,
    HardwareValidationStatus,
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_confidence as _parse_confidence,
    parse_list as _parse_list,
    parse_str as _parse_str,
    parse_str_item as _parse_str_item,
)
from sol_execbench.core.text_utils import ordered_unique


AMD_SOL_V2_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v2"
AGGREGATE_STATUSES = frozenset({"scored", "degraded", "unscored"})


@dataclass(frozen=True)
class AmdSolV2OpBound:
    """Per-operation AMD SOL bound derived from rich operator evidence."""

    node_id: str
    op_family: str
    op_name: str
    compute_bound_ms: float
    memory_bound_ms: float
    sol_bound_ms: float
    limiting_resource: str
    confidence: EstimateConfidence
    rationale: str
    estimate_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_family": self.op_family,
            "op_name": self.op_name,
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "limiting_resource": self.limiting_resource,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "estimate_warnings": list(self.estimate_warnings),
        }


@dataclass(frozen=True)
class AmdSolV2AggregateBound:
    """Aggregate bound and score eligibility state for a v2 artifact."""

    status: str
    scored: bool
    sol_bound_ms: float
    reason: str
    node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scored": self.scored,
            "sol_bound_ms": self.sol_bound_ms,
            "reason": self.reason,
            "node_ids": list(self.node_ids),
        }


@dataclass(frozen=True)
class AmdSolV2CoverageSummary:
    """Family-aware coverage summary for v2 AMD SOL bounds."""

    total_ops: int
    supported_ops: int
    inexact_ops: int
    unsupported_ops: int
    op_family_counts: dict[str, int]
    confidence_counts_by_family: dict[str, dict[str, int]]
    worst_confidence: EstimateConfidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "inexact_ops": self.inexact_ops,
            "unsupported_ops": self.unsupported_ops,
            "op_family_counts": dict(sorted(self.op_family_counts.items())),
            "confidence_counts_by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(self.confidence_counts_by_family.items())
            },
            "worst_confidence": self.worst_confidence.value,
        }


@dataclass(frozen=True)
class AmdSolBoundV2Artifact:
    """Stable AMD SOL bound artifact v2 sidecar."""

    definition: str
    workload_uuid: str
    hardware_model_ref: str | None
    hardware_model: AmdHardwareModel
    bound_graph: dict[str, object]
    operator_work_estimates: tuple[dict[str, object], ...]
    op_bounds: tuple[AmdSolV2OpBound, ...]
    aggregate_bound: AmdSolV2AggregateBound
    warnings: tuple[str, ...]
    coverage_summary: AmdSolV2CoverageSummary
    schema_version: str = AMD_SOL_V2_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model_ref": self.hardware_model_ref,
            "hardware_model": self.hardware_model.to_dict(),
            "bound_graph": dict(self.bound_graph),
            "operator_work_estimates": [
                dict(estimate) for estimate in self.operator_work_estimates
            ],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_bound": self.aggregate_bound.to_dict(),
            "warnings": list(self.warnings),
            "coverage_summary": self.coverage_summary.to_dict(),
        }


def build_amd_sol_bound_v2_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str | None = None,
) -> AmdSolBoundV2Artifact:
    """Build an AMD SOL bound artifact v2 sidecar."""
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    op_bounds = tuple(
        _bound_for_estimate(estimate, hardware_model) for estimate in estimates
    )
    coverage = _coverage_for_estimates(estimates)
    aggregate = _aggregate_for_bounds(op_bounds, hardware_model)
    warnings = _warnings_for_artifact(
        graph.warnings, estimates, aggregate, hardware_model
    )
    return AmdSolBoundV2Artifact(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model_ref=hardware_model_ref,
        hardware_model=hardware_model,
        bound_graph=graph.to_dict(),
        operator_work_estimates=tuple(estimate.to_dict() for estimate in estimates),
        op_bounds=op_bounds,
        aggregate_bound=aggregate,
        warnings=warnings,
        coverage_summary=coverage,
    )


def amd_sol_bound_v2_from_dict(payload: dict[str, Any]) -> AmdSolBoundV2Artifact:
    """Parse an AMD SOL bound artifact v2 sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("AMD SOL v2 artifact payload must be an object")
    _require_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "hardware_model_ref",
            "hardware_model",
            "bound_graph",
            "operator_work_estimates",
            "op_bounds",
            "aggregate_bound",
            "warnings",
            "coverage_summary",
        },
        source="AMD SOL v2 artifact",
    )
    schema_version = _parse_str(payload, "schema_version", source="AMD SOL v2 artifact")
    if schema_version != AMD_SOL_V2_SCHEMA_VERSION:
        raise ValueError(
            f"AMD SOL v2 artifact has invalid schema_version '{schema_version}', "
            f"expected '{AMD_SOL_V2_SCHEMA_VERSION}'"
        )

    bound_graph = _parse_dict(payload, "bound_graph", source="AMD SOL v2 artifact")
    estimate_payloads = _parse_list(
        payload, "operator_work_estimates", source="AMD SOL v2 artifact"
    )
    operator_work_estimates = tuple(
        _ensure_dict(item, source=f"operator_work_estimates[{index}]")
        for index, item in enumerate(estimate_payloads)
    )
    return AmdSolBoundV2Artifact(
        definition=_parse_str(payload, "definition", source="AMD SOL v2 artifact"),
        workload_uuid=_parse_str(
            payload, "workload_uuid", source="AMD SOL v2 artifact"
        ),
        hardware_model_ref=(
            str(payload["hardware_model_ref"])
            if payload["hardware_model_ref"] is not None
            else None
        ),
        hardware_model=amd_hardware_model_from_dict(
            _parse_dict(payload, "hardware_model", source="AMD SOL v2 artifact"),
            source="AMD SOL v2 artifact hardware_model",
        ),
        bound_graph=bound_graph,
        operator_work_estimates=operator_work_estimates,
        op_bounds=tuple(
            _op_bound_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "op_bounds", source="AMD SOL v2 artifact")
            )
        ),
        aggregate_bound=_aggregate_from_dict(
            _parse_dict(payload, "aggregate_bound", source="AMD SOL v2 artifact")
        ),
        warnings=tuple(
            _parse_str_item(item, source=f"warnings[{index}]")
            for index, item in enumerate(
                _parse_list(payload, "warnings", source="AMD SOL v2 artifact")
            )
        ),
        coverage_summary=_coverage_from_dict(
            _parse_dict(payload, "coverage_summary", source="AMD SOL v2 artifact")
        ),
        schema_version=schema_version,
        derived=bool(payload["derived"]),
    )


def _bound_for_estimate(
    estimate: OperatorWorkEstimate,
    hardware_model: AmdHardwareModel,
) -> AmdSolV2OpBound:
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
    limiting_resource = "compute" if compute_bound_ms >= memory_bound_ms else "memory"
    return AmdSolV2OpBound(
        node_id=estimate.node_id,
        op_family=estimate.op_family.value,
        op_name=estimate.op_name,
        compute_bound_ms=compute_bound_ms,
        memory_bound_ms=memory_bound_ms,
        sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
        limiting_resource=limiting_resource,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
        estimate_warnings=estimate.warnings,
    )


def _aggregate_for_bounds(
    op_bounds: tuple[AmdSolV2OpBound, ...],
    hardware_model: AmdHardwareModel,
) -> AmdSolV2AggregateBound:
    sol_bound_ms = sum(bound.sol_bound_ms for bound in op_bounds)
    node_ids = tuple(bound.node_id for bound in op_bounds)
    if not op_bounds:
        return AmdSolV2AggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="missing operation bound evidence",
            node_ids=node_ids,
        )
    if any(bound.confidence == EstimateConfidence.UNSUPPORTED for bound in op_bounds):
        return AmdSolV2AggregateBound(
            status="unscored",
            scored=False,
            sol_bound_ms=sol_bound_ms,
            reason="unsupported operation evidence present",
            node_ids=node_ids,
        )
    if (
        any(bound.confidence == EstimateConfidence.INEXACT for bound in op_bounds)
        or hardware_model.hardware_validation_status
        != HardwareValidationStatus.VALIDATED
        or hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED
        or hardware_model.confidence != EstimateConfidence.SUPPORTED
    ):
        return AmdSolV2AggregateBound(
            status="degraded",
            scored=True,
            sol_bound_ms=sol_bound_ms,
            reason="inexact or provisional evidence present",
            node_ids=node_ids,
        )
    return AmdSolV2AggregateBound(
        status="scored",
        scored=True,
        sol_bound_ms=sol_bound_ms,
        reason="all operation and hardware evidence is supported",
        node_ids=node_ids,
    )


def _coverage_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> AmdSolV2CoverageSummary:
    op_family_counts: dict[str, int] = {}
    confidence_counts: dict[str, dict[str, int]] = {}
    worst = (
        EstimateConfidence.SUPPORTED if estimates else EstimateConfidence.UNSUPPORTED
    )

    for estimate in estimates:
        family = estimate.op_family.value
        confidence = estimate.confidence.value
        op_family_counts[family] = op_family_counts.get(family, 0) + 1
        counts = confidence_counts.setdefault(
            family,
            {
                EstimateConfidence.SUPPORTED.value: 0,
                EstimateConfidence.INEXACT.value: 0,
                EstimateConfidence.UNSUPPORTED.value: 0,
            },
        )
        counts[confidence] = counts.get(confidence, 0) + 1
        worst = _worse_confidence(worst, estimate.confidence)

    return AmdSolV2CoverageSummary(
        total_ops=len(estimates),
        supported_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.SUPPORTED
        ),
        inexact_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.INEXACT
        ),
        unsupported_ops=sum(
            1
            for estimate in estimates
            if estimate.confidence == EstimateConfidence.UNSUPPORTED
        ),
        op_family_counts=op_family_counts,
        confidence_counts_by_family=confidence_counts,
        worst_confidence=worst,
    )


def _warnings_for_artifact(
    graph_warnings: tuple[str, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    aggregate: AmdSolV2AggregateBound,
    hardware_model: AmdHardwareModel,
) -> tuple[str, ...]:
    warnings: list[str] = []
    for warning in graph_warnings:
        warnings.append(f"graph_warning:{warning}")
    for estimate in estimates:
        for warning in estimate.warnings:
            warnings.append(f"estimate_warning:{estimate.node_id}:{warning}")
        if estimate.confidence == EstimateConfidence.INEXACT:
            warnings.append(
                f"inexact_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
        elif estimate.confidence == EstimateConfidence.UNSUPPORTED:
            warnings.append(
                f"unsupported_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
    if hardware_model.hardware_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "hardware_validation:"
            f"{hardware_model.architecture}:{hardware_model.hardware_validation_status.value}"
        )
    if hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(
            "model_validation:"
            f"{hardware_model.architecture}:{hardware_model.model_validation_status.value}"
        )
    if aggregate.status == "degraded":
        warnings.append(f"aggregate_degraded:{aggregate.reason}")
    elif aggregate.status == "unscored":
        warnings.append(f"aggregate_unscored:{aggregate.reason}")
    return _unique(warnings)


def _worse_confidence(
    left: EstimateConfidence,
    right: EstimateConfidence,
) -> EstimateConfidence:
    rank = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return left if rank[left] >= rank[right] else right


def _op_bound_from_dict(payload: Any, index: int) -> AmdSolV2OpBound:
    raw = _ensure_dict(payload, source=f"op_bounds[{index}]")
    _require_keys(
        raw,
        {
            "node_id",
            "op_family",
            "op_name",
            "compute_bound_ms",
            "memory_bound_ms",
            "sol_bound_ms",
            "limiting_resource",
            "confidence",
            "rationale",
            "estimate_warnings",
        },
        source=f"op_bounds[{index}]",
    )
    return AmdSolV2OpBound(
        node_id=_parse_str(raw, "node_id", source=f"op_bounds[{index}]"),
        op_family=_parse_str(raw, "op_family", source=f"op_bounds[{index}]"),
        op_name=_parse_str(raw, "op_name", source=f"op_bounds[{index}]"),
        compute_bound_ms=_parse_float(
            raw, "compute_bound_ms", source=f"op_bounds[{index}]"
        ),
        memory_bound_ms=_parse_float(
            raw, "memory_bound_ms", source=f"op_bounds[{index}]"
        ),
        sol_bound_ms=_parse_float(raw, "sol_bound_ms", source=f"op_bounds[{index}]"),
        limiting_resource=_parse_str(
            raw, "limiting_resource", source=f"op_bounds[{index}]"
        ),
        confidence=_parse_confidence(raw, "confidence", source=f"op_bounds[{index}]"),
        rationale=_parse_str(raw, "rationale", source=f"op_bounds[{index}]"),
        estimate_warnings=tuple(
            _parse_str_item(item, source=f"op_bounds[{index}].estimate_warnings[{w_i}]")
            for w_i, item in enumerate(
                _parse_list(raw, "estimate_warnings", source=f"op_bounds[{index}]")
            )
        ),
    )


def _aggregate_from_dict(payload: dict[str, Any]) -> AmdSolV2AggregateBound:
    _require_keys(
        payload,
        {"status", "scored", "sol_bound_ms", "reason", "node_ids"},
        source="aggregate_bound",
    )
    status = _parse_str(payload, "status", source="aggregate_bound")
    if status not in AGGREGATE_STATUSES:
        raise ValueError(f"aggregate_bound has invalid status '{status}'")
    return AmdSolV2AggregateBound(
        status=status,
        scored=bool(payload["scored"]),
        sol_bound_ms=_parse_float(payload, "sol_bound_ms", source="aggregate_bound"),
        reason=_parse_str(payload, "reason", source="aggregate_bound"),
        node_ids=tuple(
            _parse_str_item(item, source=f"aggregate_bound.node_ids[{index}]")
            for index, item in enumerate(
                _parse_list(payload, "node_ids", source="aggregate_bound")
            )
        ),
    )


def _coverage_from_dict(payload: dict[str, Any]) -> AmdSolV2CoverageSummary:
    _require_keys(
        payload,
        {
            "total_ops",
            "supported_ops",
            "inexact_ops",
            "unsupported_ops",
            "op_family_counts",
            "confidence_counts_by_family",
            "worst_confidence",
        },
        source="coverage_summary",
    )
    return AmdSolV2CoverageSummary(
        total_ops=_parse_int(payload, "total_ops", source="coverage_summary"),
        supported_ops=_parse_int(payload, "supported_ops", source="coverage_summary"),
        inexact_ops=_parse_int(payload, "inexact_ops", source="coverage_summary"),
        unsupported_ops=_parse_int(
            payload, "unsupported_ops", source="coverage_summary"
        ),
        op_family_counts=_parse_int_map(
            payload, "op_family_counts", source="coverage_summary"
        ),
        confidence_counts_by_family=_parse_nested_int_map(
            payload, "confidence_counts_by_family", source="coverage_summary"
        ),
        worst_confidence=_parse_confidence(
            payload, "worst_confidence", source="coverage_summary"
        ),
    )


def _require_keys(payload: dict[str, Any], required: set[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _parse_dict(payload: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = payload[key]
    return _ensure_dict(value, source=f"{source}.{key}")


def _parse_float(payload: dict[str, Any], key: str, *, source: str) -> float:
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc


def _parse_int(payload: dict[str, Any], key: str, *, source: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise ValueError(f"{source}.{key} must be an integer")
    return value


def _parse_int_map(payload: dict[str, Any], key: str, *, source: str) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, int):
            raise ValueError(f"{source}.{key}.{raw_key} must be an integer")
        parsed[raw_key] = raw_value
    return parsed


def _parse_nested_int_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, dict[str, int]]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, dict[str, int]] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, dict):
            raise ValueError(f"{source}.{key}.{raw_key} must be an object")
        parsed[raw_key] = {}
        for inner_key, inner_value in raw_value.items():
            if not isinstance(inner_key, str):
                raise ValueError(f"{source}.{key}.{raw_key} keys must be strings")
            if not isinstance(inner_value, int):
                raise ValueError(
                    f"{source}.{key}.{raw_key}.{inner_key} must be an integer"
                )
            parsed[raw_key][inner_key] = inner_value
    return parsed


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(ordered_unique(values))
