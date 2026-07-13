# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The fusion-validation-aware AMD SOL bound artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.arch_capabilities import ArchIsaBudget
from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup
from sol_execbench.core.scoring.amd_sol.builder import (
    _aggregate_for_groups,
    _build_amd_sol_bound_base,
)
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph
from sol_execbench.core.scoring.amd_sol.models import (
    AmdSolAggregateBound,
    AmdSolCoverageSummary,
    AmdSolGroupBound,
    _AmdSolBoundBase,
)
from sol_execbench.core.scoring.amd_sol.parsing import _amd_sol_bound_base_from_dict
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.fusion_validation import (
    FusionSignature,
    FusionValidationArtifact,
    fusion_validation_from_dict,
    sha256_payload,
)
from sol_execbench.core.integrity.checksums import sha256_file


AMD_SOL_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v5"
LEGACY_AMD_SOL_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v4"


@dataclass(frozen=True)
class PerformanceProviderResult:
    """A non-authoritative result supplied by one performance provider.

    A provider may describe a compiled candidate or predict its latency, but it
    cannot turn a finite compiler search space into a theoretical lower bound.
    """

    provider_name: str
    provider_revision: str
    provider_schema_version: str
    target_architecture: str
    rocm_version: str
    input_identity_sha256: str
    status: str
    result_kind: str
    is_theoretical_lower_bound: bool
    predicted_latency_ms: float | None
    measured_latency_ms: float | None
    warnings: tuple[str, ...]
    raw_evidence_ref: str | None
    raw_evidence_sha256: str | None
    output_payload: dict[str, object]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.provider_name,
                self.provider_revision,
                self.provider_schema_version,
                self.target_architecture,
                self.rocm_version,
            )
        ):
            raise ValueError("performance provider identity fields must be non-empty")
        if (
            not isinstance(self.input_identity_sha256, str)
            or (len(self.input_identity_sha256) != 64)
            or any(c not in "0123456789abcdef" for c in self.input_identity_sha256)
        ):
            raise ValueError("provider input identity must be a lowercase SHA-256")
        if self.status not in {"supported", "inexact", "unavailable", "failed"}:
            raise ValueError("performance provider status is invalid")
        if self.result_kind not in {
            "prediction",
            "compiled_candidate",
            "measurement",
        }:
            raise ValueError("performance provider result_kind is invalid")
        if not isinstance(self.is_theoretical_lower_bound, bool):
            raise ValueError("performance provider lower-bound flag must be a boolean")
        if self.is_theoretical_lower_bound:
            raise ValueError(
                "external provider results cannot claim a theoretical lower bound"
            )
        for name, value in (
            ("predicted_latency_ms", self.predicted_latency_ms),
            ("measured_latency_ms", self.measured_latency_ms),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number or null")
        if not all(isinstance(warning, str) for warning in self.warnings):
            raise ValueError("performance provider warnings must be strings")
        if (self.raw_evidence_ref is None) != (self.raw_evidence_sha256 is None):
            raise ValueError("raw performance evidence ref and checksum must be paired")
        if self.raw_evidence_ref is not None and (
            not isinstance(self.raw_evidence_ref, str)
            or not self.raw_evidence_ref.strip()
        ):
            raise ValueError("raw performance evidence ref must be a non-empty string")
        if self.raw_evidence_sha256 is not None and (
            not isinstance(self.raw_evidence_sha256, str)
            or len(self.raw_evidence_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.raw_evidence_sha256)
        ):
            raise ValueError(
                "raw performance evidence checksum must be a lowercase SHA-256"
            )
        if (
            self.predicted_latency_ms is not None
            or self.measured_latency_ms is not None
        ) and self.raw_evidence_ref is None:
            raise ValueError("performance latency requires raw evidence provenance")
        if not isinstance(self.output_payload, dict):
            raise ValueError("performance provider output_payload must be an object")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "provider_revision": self.provider_revision,
            "provider_schema_version": self.provider_schema_version,
            "target_architecture": self.target_architecture,
            "rocm_version": self.rocm_version,
            "input_identity_sha256": self.input_identity_sha256,
            "status": self.status,
            "result_kind": self.result_kind,
            "is_theoretical_lower_bound": self.is_theoretical_lower_bound,
            "predicted_latency_ms": self.predicted_latency_ms,
            "measured_latency_ms": self.measured_latency_ms,
            "warnings": list(self.warnings),
            "raw_evidence_ref": self.raw_evidence_ref,
            "raw_evidence_sha256": self.raw_evidence_sha256,
            "output_payload": dict(self.output_payload),
        }


@dataclass(frozen=True)
class AmdSolPerformanceDiagnostics:
    """Diagnostic predictions and observations, deliberately outside authority."""

    provider_results: tuple[PerformanceProviderResult, ...] = ()

    @classmethod
    def from_provider_results(
        cls, provider_results: tuple[PerformanceProviderResult, ...]
    ) -> "AmdSolPerformanceDiagnostics":
        return cls(provider_results=provider_results)

    @property
    def status(self) -> str:
        if not self.provider_results:
            return "not_requested"
        if any(
            result.predicted_latency_ms is not None
            or result.measured_latency_ms is not None
            for result in self.provider_results
        ):
            return "available"
        if any(result.status == "failed" for result in self.provider_results):
            return "failed"
        if any(result.status == "inexact" for result in self.provider_results):
            return "inexact"
        return "unavailable"

    @property
    def t_predicted_best_ms(self) -> float | None:
        values = [
            result.predicted_latency_ms
            for result in self.provider_results
            if result.predicted_latency_ms is not None
        ]
        return min(values) if values else None

    @property
    def fastest_known_ms(self) -> float | None:
        values = [
            result.measured_latency_ms
            for result in self.provider_results
            if result.status == "supported"
            and result.result_kind in {"compiled_candidate", "measurement"}
            and result.measured_latency_ms is not None
        ]
        return min(values) if values else None

    def to_dict(self, *, t_sol_floor_ms: float) -> dict[str, object]:
        fastest_known_ms = self.fastest_known_ms
        predicted_best_ms = self.t_predicted_best_ms
        return {
            "status": self.status,
            "t_predicted_best_ms": predicted_best_ms,
            "fastest_known_ms": fastest_known_ms,
            "t_sol_floor_to_fastest_known_ratio": (
                t_sol_floor_ms / fastest_known_ms
                if fastest_known_ms is not None
                else None
            ),
            "t_predicted_best_to_fastest_known_ratio": (
                predicted_best_ms / fastest_known_ms
                if predicted_best_ms is not None and fastest_known_ms is not None
                else None
            ),
            "floor_contradicts_fastest_known": (
                fastest_known_ms is not None and t_sol_floor_ms > fastest_known_ms
            ),
            "provider_results": [result.to_dict() for result in self.provider_results],
        }


@dataclass(frozen=True)
class FusionGroupEvidenceSummary:
    group_id: str
    signature_sha256: str
    variant_id: str | None
    capacity_evidence_id: str | None
    capacity_status: str
    performance_status: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AmdSolBoundArtifact:
    """AMD SOL artifact with separate authority and diagnostic tracks."""

    base: _AmdSolBoundBase
    fusion_validation_ref: str
    fusion_validation_sha256: str
    fusion_validation_matches: tuple[FusionGroupEvidenceSummary, ...]
    performance_diagnostics: AmdSolPerformanceDiagnostics = (
        AmdSolPerformanceDiagnostics()
    )
    schema_version: str = AMD_SOL_SCHEMA_VERSION

    @property
    def definition(self) -> str:
        return self.base.definition

    @property
    def workload_uuid(self) -> str:
        return self.base.workload_uuid

    @property
    def hardware_model_ref(self) -> str | None:
        return self.base.hardware_model_ref

    @property
    def hardware_model(self) -> AmdHardwareModel:
        return self.base.hardware_model

    @property
    def capability_budget_ref(self) -> str | None:
        return self.base.capability_budget_ref

    @property
    def capability_budget(self) -> ArchIsaBudget | None:
        return self.base.capability_budget

    @property
    def bound_graph(self) -> dict[str, object]:
        return self.base.bound_graph

    @property
    def operator_work_estimates(self) -> tuple[dict[str, object], ...]:
        return self.base.operator_work_estimates

    @property
    def fusion_groups(self) -> tuple[FusionGroup, ...]:
        return self.base.fusion_groups

    @property
    def group_bounds(self) -> tuple[AmdSolGroupBound, ...]:
        return self.base.group_bounds

    @property
    def aggregate_bound(self) -> AmdSolAggregateBound:
        return self.base.aggregate_bound

    @property
    def t_sol_floor_ms(self) -> float:
        """The only latency value eligible to be consumed as a SOL authority."""
        return self.base.aggregate_bound.sol_bound_ms

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.base.warnings

    @property
    def coverage_summary(self) -> AmdSolCoverageSummary:
        return self.base.coverage_summary

    @property
    def derived(self) -> bool:
        return self.base.derived

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version == LEGACY_AMD_SOL_SCHEMA_VERSION:
            return self._to_v4_dict()
        if self.schema_version != AMD_SOL_SCHEMA_VERSION:
            raise ValueError("AMD SOL artifact has invalid schema_version")
        payload = self.base.to_dict()
        payload["schema_version"] = self.schema_version
        aggregate = payload.pop("aggregate_bound")
        payload["theoretical_lower_bound"] = {
            "status": aggregate["status"],
            "authority_eligible": aggregate["scored"],
            "t_sol_floor_ms": aggregate["sol_bound_ms"],
            "reason": aggregate["reason"],
            "node_ids": aggregate["node_ids"],
        }
        payload["group_bounds"] = [
            {
                **{key: value for key, value in bound.items() if key != "sol_bound_ms"},
                "t_sol_floor_ms": bound["sol_bound_ms"],
            }
            for bound in payload["group_bounds"]
        ]
        summaries = {item.group_id: item for item in self.fusion_validation_matches}
        groups = []
        for group in payload["fusion_groups"]:
            summary = summaries[group["group_id"]]
            groups.append(
                {
                    **group,
                    "signature_sha256": summary.signature_sha256,
                    "variant_id": summary.variant_id,
                    "capacity_evidence_id": summary.capacity_evidence_id,
                    "capacity_status": summary.capacity_status,
                    "performance_status": summary.performance_status,
                }
            )
        payload["fusion_groups"] = groups
        payload["fusion_validation_ref"] = self.fusion_validation_ref
        payload["fusion_validation_sha256"] = self.fusion_validation_sha256
        payload["fusion_validation_matches"] = [
            item.to_dict() for item in self.fusion_validation_matches
        ]
        payload["performance_diagnostics"] = self.performance_diagnostics.to_dict(
            t_sol_floor_ms=self.t_sol_floor_ms
        )
        return payload

    def _to_v4_dict(self) -> dict[str, Any]:
        """Serialize a parsed legacy artifact without changing its field meanings."""
        payload = self.base.to_dict()
        payload["schema_version"] = LEGACY_AMD_SOL_SCHEMA_VERSION
        summaries = {item.group_id: item for item in self.fusion_validation_matches}
        payload["fusion_groups"] = [
            {
                **group,
                "signature_sha256": summaries[group["group_id"]].signature_sha256,
                "variant_id": summaries[group["group_id"]].variant_id,
                "capacity_evidence_id": summaries[
                    group["group_id"]
                ].capacity_evidence_id,
                "capacity_status": summaries[group["group_id"]].capacity_status,
                "performance_status": summaries[group["group_id"]].performance_status,
            }
            for group in payload["fusion_groups"]
        ]
        payload["fusion_validation_ref"] = self.fusion_validation_ref
        payload["fusion_validation_sha256"] = self.fusion_validation_sha256
        payload["fusion_validation_matches"] = [
            item.to_dict() for item in self.fusion_validation_matches
        ]
        return payload


def fusion_signature_for_group(
    group: FusionGroup,
    bound_graph: dict[str, object],
    *,
    tile_contract: dict[str, object] | None = None,
) -> FusionSignature:
    """Construct the canonical signature from graph order and exact tensors."""
    raw_nodes = bound_graph.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError("bound graph nodes must be a list")
    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue
        node = cast(dict[str, Any], raw_node)
        if isinstance(node.get("node_id"), str):
            nodes[node["node_id"]] = node
    raw_tensors = bound_graph.get("tensors", {})
    if not isinstance(raw_tensors, dict):
        raise ValueError("bound graph tensors must be an object")
    tensors = cast(dict[str, Any], raw_tensors)

    def shape(tensor_id: str) -> tuple[int, ...]:
        tensor = tensors.get(tensor_id)
        raw = tensor.get("shape") if isinstance(tensor, dict) else None
        if not isinstance(raw, list) or not all(
            isinstance(dim, int) and dim > 0 for dim in raw
        ):
            raise ValueError(f"fusion tensor {tensor_id} has no exact static shape")
        return tuple(raw)

    op_names = tuple(str(nodes[node_id]["op_name"]) for node_id in group.node_ids)
    dtypes = {
        str(tensors[tensor_id]["dtype"])
        for tensor_id in (
            *group.external_input_tensor_ids,
            *group.external_output_tensor_ids,
        )
        if isinstance(tensors.get(tensor_id), dict)
    }
    if len(dtypes) != 1:
        raise ValueError("fusion signature requires one exact dtype")
    contract: dict[str, object] = tile_contract or {
        "required_lds_bytes": group.required_lds_bytes
    }
    return FusionSignature(
        pattern_id=group.pattern_id,
        pattern_version=group.pattern_version,
        op_names=op_names,
        dtype=next(iter(dtypes)),
        input_shapes=tuple(shape(item) for item in group.external_input_tensor_ids),
        output_shapes=tuple(shape(item) for item in group.external_output_tensor_ids),
        tile_contract=contract,
    )


def _matching_evidence_tile_contract(
    group: FusionGroup,
    bound_graph: dict[str, object],
    evidence: FusionValidationArtifact,
    *,
    workload_uuid: str,
) -> dict[str, object] | None:
    """Return a uniquely identified evidence contract for one fusion group."""
    if len(group.node_ids) <= 1:
        return None
    try:
        signature = fusion_signature_for_group(group, bound_graph)
    except ValueError:
        return None
    matches = [
        case
        for case in evidence.cases
        if case.workload_uuid == workload_uuid
        and case.signature.pattern_id == signature.pattern_id
        and case.signature.pattern_version == signature.pattern_version
        and case.signature.op_names == signature.op_names
        and case.signature.dtype == signature.dtype
        and case.signature.input_shapes == signature.input_shapes
        and case.signature.output_shapes == signature.output_shapes
    ]
    if len(matches) != 1:
        return None
    return dict(matches[0].signature.tile_contract)


def build_amd_sol_bound_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    fusion_validation: FusionValidationArtifact | dict[str, Any],
    fusion_validation_ref: str,
    fusion_validation_sha256: str,
    evidence_path: Path | None = None,
    hardware_model_ref: str | None = None,
    capability_budget_ref: str | None = None,
    capability_budget: ArchIsaBudget | None = None,
    tile_contracts: dict[str, dict[str, object]] | None = None,
    bound_graph: BoundGraph | None = None,
    performance_diagnostics: AmdSolPerformanceDiagnostics | None = None,
) -> AmdSolBoundArtifact:
    """Build the authority floor without allowing diagnostics to influence it."""
    evidence_payload = (
        cast(dict[str, Any], fusion_validation)
        if isinstance(fusion_validation, dict)
        else cast(dict[str, Any], fusion_validation.to_dict())
    )
    evidence = fusion_validation_from_dict(evidence_payload)
    if not fusion_validation_ref.strip():
        raise ValueError("fusion_validation_ref must be a non-empty path")
    if evidence.architecture != hardware_model.architecture.lower():
        raise ValueError("fusion evidence architecture does not match hardware model")
    if (
        evidence_path is not None
        and sha256_file(evidence_path) != fusion_validation_sha256
    ):
        raise ValueError("fusion evidence file checksum mismatch")
    if len(fusion_validation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fusion_validation_sha256
    ):
        raise ValueError("fusion_validation_sha256 must be a SHA-256")
    diagnostics = performance_diagnostics or AmdSolPerformanceDiagnostics()
    _validate_performance_diagnostics(diagnostics, hardware_model.architecture)
    base = _build_amd_sol_bound_base(
        definition,
        workload,
        hardware_model,
        hardware_model_ref=hardware_model_ref,
        capability_budget_ref=capability_budget_ref,
        capability_budget=capability_budget,
        bound_graph=bound_graph,
    )
    summaries: list[FusionGroupEvidenceSummary] = []
    promoted_groups: list[FusionGroup] = []
    promoted_bounds: list[AmdSolGroupBound] = []
    missing_validation_group_ids: list[str] = []
    unproven_singleton_partition = len(base.fusion_groups) > 1 and all(
        group.pattern_id == "singleton.v1" for group in base.fusion_groups
    )
    estimates = {item["node_id"]: item for item in base.operator_work_estimates}
    for group, bound in zip(base.fusion_groups, base.group_bounds, strict=True):
        tile_contract = (tile_contracts or {}).get(group.group_id)
        if tile_contract is None:
            tile_contract = _matching_evidence_tile_contract(
                group,
                base.bound_graph,
                evidence,
                workload_uuid=str(workload.uuid),
            )
        try:
            signature = fusion_signature_for_group(
                group,
                base.bound_graph,
                tile_contract=tile_contract,
            )
        except ValueError:
            signature = None
        case = (
            evidence.matching_case(signature, workload_uuid=workload.uuid)
            if signature is not None
            else None
        )
        passed = case is not None and case.capacity_status == "passed"
        summaries.append(
            FusionGroupEvidenceSummary(
                group_id=group.group_id,
                signature_sha256=(
                    signature.canonical_id
                    if signature is not None
                    else sha256_payload({"unmatchable_fusion_group": group.to_dict()})
                ),
                variant_id=case.variant_id if case else None,
                capacity_evidence_id=case.evidence_id if passed else None,
                capacity_status="passed" if passed else "missing",
                performance_status=case.performance.status if case else "not_measured",
            )
        )
        if passed and "fusion_capacity_evidence_missing" in group.warnings:
            group_warnings = tuple(
                item
                for item in group.warnings
                if item != "fusion_capacity_evidence_missing"
            )
            group_confidence = (
                EstimateConfidence.SUPPORTED
                if not group_warnings
                and all(
                    estimates[node_id].get("confidence") == "supported"
                    for node_id in group.node_ids
                )
                else group.confidence
            )
            group = replace(group, warnings=group_warnings, confidence=group_confidence)
            bound_warnings = tuple(
                item
                for item in bound.warnings
                if item != "fusion_capacity_evidence_missing"
            )
            bound_confidence = (
                EstimateConfidence.SUPPORTED
                if group_confidence == EstimateConfidence.SUPPORTED
                and not bound_warnings
                else bound.confidence
            )
            bound = replace(bound, warnings=bound_warnings, confidence=bound_confidence)
        elif len(group.node_ids) > 1 and not passed:
            missing_validation_group_ids.append(group.group_id)
            missing_warning = "fusion_validation_evidence_missing"
            group = replace(
                group,
                confidence=(
                    EstimateConfidence.UNSUPPORTED
                    if group.confidence == EstimateConfidence.UNSUPPORTED
                    else EstimateConfidence.INEXACT
                ),
                warnings=tuple(dict.fromkeys((*group.warnings, missing_warning))),
            )
            bound = replace(
                bound,
                confidence=(
                    EstimateConfidence.UNSUPPORTED
                    if bound.confidence == EstimateConfidence.UNSUPPORTED
                    else EstimateConfidence.INEXACT
                ),
                warnings=tuple(dict.fromkeys((*bound.warnings, missing_warning))),
            )
        if unproven_singleton_partition:
            # A registry miss is not a semantic materialization barrier.  Summing
            # singleton rooflines would turn the current compiler partition into
            # a purported lower bound that a longer fused implementation can beat.
            warning = "unproven_multinode_singleton_partition"
            group = replace(
                group,
                confidence=EstimateConfidence.UNSUPPORTED,
                warnings=tuple(dict.fromkeys((*group.warnings, warning))),
            )
            bound = replace(
                bound,
                confidence=EstimateConfidence.UNSUPPORTED,
                warnings=tuple(dict.fromkeys((*bound.warnings, warning))),
            )
        promoted_groups.append(group)
        promoted_bounds.append(bound)
    groups_tuple, bounds_tuple = tuple(promoted_groups), tuple(promoted_bounds)
    aggregate = _aggregate_for_groups(bounds_tuple, hardware_model)
    if (
        diagnostics.fastest_known_ms is not None
        and aggregate.sol_bound_ms > diagnostics.fastest_known_ms
    ):
        # A validated, faster implementation disproves a claimed lower bound.
        # This is an authority failure, not merely a diagnostic warning.
        aggregate = replace(
            aggregate,
            status="unscored",
            scored=False,
            reason="theoretical floor contradicts fastest-known measurement",
        )
    passed_group_ids = {
        item.group_id for item in summaries if item.capacity_status == "passed"
    }
    warnings = tuple(
        warning
        for warning in base.warnings
        if not (
            "fusion_capacity_evidence_missing" in warning
            and any(f":{group_id}:" in warning for group_id in passed_group_ids)
        )
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *warnings,
                *(
                    f"fusion_validation_evidence_missing:{group_id}"
                    for group_id in missing_validation_group_ids
                ),
                *(
                    f"fusion_warning:{group.group_id}:{warning}"
                    for group in groups_tuple
                    for warning in group.warnings
                ),
                *(
                    f"fusion_bound_warning:{bound.group_id}:{warning}"
                    for bound in bounds_tuple
                    for warning in bound.warnings
                ),
                *(
                    ("floor_contradicts_fastest_known",)
                    if diagnostics.fastest_known_ms is not None
                    and aggregate.sol_bound_ms > diagnostics.fastest_known_ms
                    else ()
                ),
                *(
                    (f"aggregate_{aggregate.status}:{aggregate.reason}",)
                    if aggregate.status != "scored"
                    else ()
                ),
            )
        )
    )
    if aggregate.status == "scored":
        warnings = tuple(w for w in warnings if not w.startswith("aggregate_degraded:"))
    base = replace(
        base,
        fusion_groups=groups_tuple,
        group_bounds=bounds_tuple,
        aggregate_bound=aggregate,
        warnings=warnings,
    )
    return AmdSolBoundArtifact(
        base=base,
        fusion_validation_ref=fusion_validation_ref,
        fusion_validation_sha256=fusion_validation_sha256,
        fusion_validation_matches=tuple(summaries),
        performance_diagnostics=diagnostics,
    )


def amd_sol_bound_from_dict(payload: dict[str, Any]) -> AmdSolBoundArtifact:
    """Strictly parse v5 dual-track artifacts and preserve legacy v4 payloads."""
    schema_version = payload.get("schema_version")
    if schema_version == LEGACY_AMD_SOL_SCHEMA_VERSION:
        return _amd_sol_bound_from_v4_dict(payload)
    if schema_version != AMD_SOL_SCHEMA_VERSION:
        raise ValueError("AMD SOL artifact has invalid schema_version")
    return _amd_sol_bound_from_v5_dict(payload)


def _amd_sol_bound_from_v5_dict(payload: dict[str, Any]) -> AmdSolBoundArtifact:
    required_extra = {
        "fusion_validation_ref",
        "fusion_validation_sha256",
        "fusion_validation_matches",
        "performance_diagnostics",
        "theoretical_lower_bound",
    }
    base_keys = set(_AmdSolBoundBase.__dataclass_fields__) - {
        "derived",
        "aggregate_bound",
    }
    expected = base_keys | {"derived", "schema_version"} | required_extra
    if set(payload) != expected:
        raise ValueError("AMD SOL v5 artifact has missing or unknown top-level fields")
    floor = payload["theoretical_lower_bound"]
    if not isinstance(floor, dict) or set(floor) != {
        "status",
        "authority_eligible",
        "t_sol_floor_ms",
        "reason",
        "node_ids",
    }:
        raise ValueError("theoretical_lower_bound has missing or unknown fields")
    if not isinstance(floor["authority_eligible"], bool):
        raise ValueError("theoretical_lower_bound.authority_eligible must be a boolean")
    raw_bounds = payload["group_bounds"]
    if not isinstance(raw_bounds, list):
        raise ValueError("group_bounds must be a list")
    legacy_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"theoretical_lower_bound", "performance_diagnostics"}
    }
    legacy_payload["schema_version"] = LEGACY_AMD_SOL_SCHEMA_VERSION
    legacy_payload["aggregate_bound"] = {
        "status": floor["status"],
        "scored": floor["authority_eligible"],
        "sol_bound_ms": floor["t_sol_floor_ms"],
        "reason": floor["reason"],
        "node_ids": floor["node_ids"],
    }
    legacy_payload["group_bounds"] = [
        {
            **{key: value for key, value in bound.items() if key != "t_sol_floor_ms"},
            "sol_bound_ms": bound.get("t_sol_floor_ms"),
        }
        if isinstance(bound, dict)
        else bound
        for bound in raw_bounds
    ]
    legacy = _amd_sol_bound_from_v4_dict(legacy_payload)
    diagnostics = _performance_diagnostics_from_dict(
        payload["performance_diagnostics"], t_sol_floor_ms=legacy.t_sol_floor_ms
    )
    _validate_performance_diagnostics(diagnostics, legacy.hardware_model.architecture)
    return replace(
        legacy,
        performance_diagnostics=diagnostics,
        schema_version=AMD_SOL_SCHEMA_VERSION,
    )


def _amd_sol_bound_from_v4_dict(payload: dict[str, Any]) -> AmdSolBoundArtifact:
    """Parse v4 exactly; it has no diagnostic track and retains old names."""
    required_extra = {
        "fusion_validation_ref",
        "fusion_validation_sha256",
        "fusion_validation_matches",
    }
    v3_keys = set(_AmdSolBoundBase.__dataclass_fields__) - {"derived"}
    v3_keys.add("derived")
    expected = v3_keys | {"schema_version"} | required_extra
    if set(payload) != expected:
        raise ValueError("AMD SOL v4 artifact has missing or unknown top-level fields")
    if payload["schema_version"] != LEGACY_AMD_SOL_SCHEMA_VERSION:
        raise ValueError("AMD SOL artifact has invalid schema_version")
    if (
        not isinstance(payload["fusion_validation_ref"], str)
        or not payload["fusion_validation_ref"]
    ):
        raise ValueError("fusion_validation_ref must be a non-empty string")
    checksum = payload["fusion_validation_sha256"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(c not in "0123456789abcdef" for c in checksum)
    ):
        raise ValueError("fusion_validation_sha256 must be a lowercase SHA-256")
    raw_groups = payload["fusion_groups"]
    if not isinstance(raw_groups, list):
        raise ValueError("fusion_groups must be a list")
    extra_group_keys = {
        "signature_sha256",
        "variant_id",
        "capacity_evidence_id",
        "capacity_status",
        "performance_status",
    }
    stripped_groups = []
    embedded = []
    for group in raw_groups:
        if not isinstance(group, dict) or not extra_group_keys.issubset(group):
            raise ValueError("v4 fusion group validation fields are missing")
        embedded.append(
            {
                "group_id": group.get("group_id"),
                **{key: group[key] for key in extra_group_keys},
            }
        )
        stripped_groups.append(
            {key: value for key, value in group.items() if key not in extra_group_keys}
        )
    v3_payload = {
        key: value for key, value in payload.items() if key not in required_extra
    }
    v3_payload["schema_version"] = "sol_execbench.amd_sol_bound.v3"
    v3_payload["fusion_groups"] = stripped_groups
    base = _amd_sol_bound_base_from_dict(v3_payload)
    raw_matches = payload["fusion_validation_matches"]
    if not isinstance(raw_matches, list) or embedded != raw_matches:
        raise ValueError("fusion_validation_matches must exactly mirror fusion groups")
    matches = tuple(_summary(item) for item in raw_matches)
    return AmdSolBoundArtifact(
        base,
        payload["fusion_validation_ref"],
        checksum,
        matches,
        schema_version=LEGACY_AMD_SOL_SCHEMA_VERSION,
    )


def _performance_diagnostics_from_dict(
    raw: object, *, t_sol_floor_ms: float
) -> AmdSolPerformanceDiagnostics:
    expected = {
        "status",
        "t_predicted_best_ms",
        "fastest_known_ms",
        "t_sol_floor_to_fastest_known_ratio",
        "t_predicted_best_to_fastest_known_ratio",
        "floor_contradicts_fastest_known",
        "provider_results",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("performance_diagnostics has missing or unknown fields")
    raw = cast(dict[str, Any], raw)
    results_raw = raw["provider_results"]
    if not isinstance(results_raw, list):
        raise ValueError("performance_diagnostics.provider_results must be a list")
    diagnostics = AmdSolPerformanceDiagnostics(
        tuple(_performance_provider_result_from_dict(item) for item in results_raw)
    )
    expected_payload = diagnostics.to_dict(t_sol_floor_ms=t_sol_floor_ms)
    if raw != expected_payload:
        raise ValueError(
            "performance_diagnostics values must be derived from providers"
        )
    return diagnostics


def _performance_provider_result_from_dict(raw: object) -> PerformanceProviderResult:
    expected = set(PerformanceProviderResult.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("performance provider result has missing or unknown fields")
    raw = cast(dict[str, Any], raw)
    string_keys = {
        "provider_name",
        "provider_revision",
        "provider_schema_version",
        "target_architecture",
        "rocm_version",
        "input_identity_sha256",
        "status",
        "result_kind",
    }
    if any(not isinstance(raw[key], str) for key in string_keys):
        raise ValueError(
            "performance provider identity and state fields must be strings"
        )
    if not isinstance(raw["is_theoretical_lower_bound"], bool):
        raise ValueError("performance provider lower-bound flag must be a boolean")
    for key in ("predicted_latency_ms", "measured_latency_ms"):
        if raw[key] is not None and (
            isinstance(raw[key], bool) or not isinstance(raw[key], int | float)
        ):
            raise ValueError(f"performance provider {key} must be a number or null")
    if not isinstance(raw["warnings"], list) or not all(
        isinstance(item, str) for item in raw["warnings"]
    ):
        raise ValueError("performance provider warnings must be a string list")
    for key in ("raw_evidence_ref", "raw_evidence_sha256"):
        if raw[key] is not None and not isinstance(raw[key], str):
            raise ValueError(f"performance provider {key} must be a string or null")
    if not isinstance(raw["output_payload"], dict):
        raise ValueError("performance provider output_payload must be an object")
    return PerformanceProviderResult(
        provider_name=raw["provider_name"],
        provider_revision=raw["provider_revision"],
        provider_schema_version=raw["provider_schema_version"],
        target_architecture=raw["target_architecture"],
        rocm_version=raw["rocm_version"],
        input_identity_sha256=raw["input_identity_sha256"],
        status=raw["status"],
        result_kind=raw["result_kind"],
        is_theoretical_lower_bound=raw["is_theoretical_lower_bound"],
        predicted_latency_ms=raw["predicted_latency_ms"],
        measured_latency_ms=raw["measured_latency_ms"],
        warnings=tuple(raw["warnings"]),
        raw_evidence_ref=raw["raw_evidence_ref"],
        raw_evidence_sha256=raw["raw_evidence_sha256"],
        output_payload=dict(raw["output_payload"]),
    )


def _validate_performance_diagnostics(
    diagnostics: AmdSolPerformanceDiagnostics, architecture: str
) -> None:
    if any(
        result.target_architecture.lower() != architecture.lower()
        for result in diagnostics.provider_results
    ):
        raise ValueError(
            "performance provider target architecture does not match hardware model"
        )


def _summary(raw: dict[str, Any]) -> FusionGroupEvidenceSummary:
    expected = set(FusionGroupEvidenceSummary.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("fusion validation match has unknown or missing fields")
    if raw["capacity_status"] not in {"passed", "missing"}:
        raise ValueError("fusion validation match has invalid capacity_status")
    if raw["performance_status"] not in {
        "passed",
        "failed",
        "unstable",
        "not_measured",
    }:
        raise ValueError("fusion validation match has invalid performance_status")
    if raw["capacity_status"] == "passed" and not raw["capacity_evidence_id"]:
        raise ValueError("passed capacity match requires capacity_evidence_id")
    for key in ("group_id", "signature_sha256"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"fusion validation match {key} is invalid")
    checksum = raw["signature_sha256"]
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("fusion validation match signature_sha256 is invalid")
    return FusionGroupEvidenceSummary(**raw)


__all__ = [
    "AMD_SOL_SCHEMA_VERSION",
    "LEGACY_AMD_SOL_SCHEMA_VERSION",
    "AmdSolBoundArtifact",
    "AmdSolPerformanceDiagnostics",
    "FusionGroupEvidenceSummary",
    "PerformanceProviderResult",
    "amd_sol_bound_from_dict",
    "build_amd_sol_bound_artifact",
    "fusion_signature_for_group",
]
