# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The fusion-validation-aware AMD SOL bound artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


AMD_SOL_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v4"


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
    """AMD SOL payload with independently bound fusion-validation evidence."""

    base: _AmdSolBoundBase
    fusion_validation_ref: str
    fusion_validation_sha256: str
    fusion_validation_matches: tuple[FusionGroupEvidenceSummary, ...]
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
    def warnings(self) -> tuple[str, ...]:
        return self.base.warnings

    @property
    def coverage_summary(self) -> AmdSolCoverageSummary:
        return self.base.coverage_summary

    @property
    def derived(self) -> bool:
        return self.base.derived

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload["schema_version"] = self.schema_version
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
) -> AmdSolBoundArtifact:
    """Build the bound, promoting only uniquely matched, capacity-passed groups."""
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
        promoted_groups.append(group)
        promoted_bounds.append(bound)
    groups_tuple, bounds_tuple = tuple(promoted_groups), tuple(promoted_bounds)
    aggregate = _aggregate_for_groups(bounds_tuple, hardware_model)
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
    )


def amd_sol_bound_from_dict(payload: dict[str, Any]) -> AmdSolBoundArtifact:
    """Strict parser for the sole supported AMD SOL schema."""
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
    if payload["schema_version"] != AMD_SOL_SCHEMA_VERSION:
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
        base, payload["fusion_validation_ref"], checksum, matches
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
    "AmdSolBoundArtifact",
    "FusionGroupEvidenceSummary",
    "amd_sol_bound_from_dict",
    "build_amd_sol_bound_artifact",
    "fusion_signature_for_group",
]
