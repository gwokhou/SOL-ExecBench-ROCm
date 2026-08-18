# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Agent projections derived from complete evaluator target views."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusTargetViewManifest,
    GeneratedWorkloadRecord,
    GenerationDecision,
    GenerationDecisionStatus,
    WorkloadRole,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.generalization.models import (
    AgentDefinitionView,
    CorpusAgentView,
    HardwareContextView,
    HardwareShift,
    NormalizedHardwareFacts,
    TrainingExposureDeclaration,
)
from sol_execbench.core.integrity import stable_json_checksum


def classify_hardware_shift(
    exposure: TrainingExposureDeclaration,
    target_view: CorpusTargetViewManifest,
) -> HardwareShift:
    """Derive the shift label from declared training exposure."""
    gfx = target_view.target.gfx_target.strip().lower()
    same_distribution = tuple(
        item
        for item in exposure.hardware
        if item.distribution_id == target_view.distribution_id
    )
    if exposure.hardware and not same_distribution:
        raise ValueError(
            "training exposure uses a different workload distribution"
        )
    same_arch = tuple(
        item
        for item in same_distribution
        if item.gfx_target.strip().lower() == gfx
    )
    if not same_arch and any(
        item.gfx_target.strip().lower() == gfx for item in exposure.hardware
    ):
        raise ValueError(
            "same-ISA training exposure uses a different workload distribution"
        )
    if not same_arch:
        return HardwareShift.UNSEEN_ARCHITECTURE
    exact = any(
        item.hardware_configuration_id == target_view.hardware_configuration_id
        and item.capacity_class_bytes == target_view.capacity_class_bytes
        for item in same_arch
    )
    if exact:
        return HardwareShift.SEEN_CONFIGURATION
    same_capacity = any(
        item.capacity_class_bytes == target_view.capacity_class_bytes
        for item in same_arch
    )
    if same_capacity:
        return HardwareShift.SAME_ISA_NEW_CONFIGURATION
    return HardwareShift.SAME_ISA_NEW_CAPACITY


def hardware_facts(
    target_view: CorpusTargetViewManifest,
    *,
    study_target_id: str,
    anonymous: bool = False,
) -> NormalizedHardwareFacts:
    """Project only facts already used by workload generation."""
    target = target_view.target
    configuration = target_view.hardware_context.configuration
    expose_identity = not anonymous
    payload: dict[str, object] = {
        "study_target_id": study_target_id,
        "context_view": (
            HardwareContextView.ANONYMIZED_FACTS
            if anonymous
            else HardwareContextView.FULL_FACTS
        ),
        "gfx_target": (
            target.gfx_target.strip().lower() if expose_identity else None
        ),
        "target_id": target.target_id if expose_identity else None,
        "hardware_configuration_id": (
            target_view.hardware_configuration_id if expose_identity else None
        ),
        "device_model": configuration.device_model if expose_identity else None,
        "product_sku": configuration.product_sku if expose_identity else None,
        "configuration_kind": (configuration.kind if expose_identity else None),
        "visible_compute_units": (
            configuration.visible_compute_units if expose_identity else None
        ),
        "visible_memory_bytes": (
            configuration.visible_memory_bytes if expose_identity else None
        ),
        "partition": configuration.partition if expose_identity else None,
        "virtualization": (
            configuration.virtualization if expose_identity else None
        ),
        "isolation": configuration.isolation if expose_identity else None,
        "capacity_class_bytes": target_view.capacity_class_bytes,
        "supported_dtypes": sorted(set(map(str, target.supported_dtypes))),
        "supported_quantization": sorted(
            set(map(str, target.supported_quantization))
        ),
        "capabilities": sorted(set(map(str, target.capabilities))),
        "max_tensor_bytes": target.max_tensor_bytes,
        "reference_ipc_limit_bytes": target.reference_ipc_limit_bytes,
    }
    payload["context_digest"] = stable_json_checksum(payload)
    return NormalizedHardwareFacts.model_validate(payload)


def build_agent_view(
    manifest: CorpusManifest,
    target_view: CorpusTargetViewManifest,
    facts: NormalizedHardwareFacts,
) -> CorpusAgentView:
    """Remove concrete smoke and holdout workloads from an Agent view."""
    _validate_facts(target_view, facts)
    entries = {entry.semantic_id: entry for entry in manifest.entries}
    decisions = {item.semantic_id: item for item in target_view.decisions}
    records: dict[str, list[GeneratedWorkloadRecord]] = defaultdict(list)
    for workload in target_view.workloads:
        records[workload.semantic_id].append(workload)
    definitions = tuple(
        _definition_view(
            entries[semantic_id],
            decisions[semantic_id],
            records[semantic_id],
        )
        for semantic_id in sorted(records)
    )
    payload: dict[str, object] = {
        "schema_version": DatasetArtifactSchema.CORPUS_AGENT_VIEW,
        "corpus_id": target_view.corpus_id,
        "release_id": target_view.release_id,
        "source_manifest_sha256": target_view.source_manifest_sha256,
        "hardware_facts": facts,
        "requested_profiles": target_view.requested_profiles,
        "public_generator_version": target_view.generator_version,
        "definitions": definitions,
    }
    payload["agent_view_digest"] = stable_json_checksum(_json_payload(payload))
    return CorpusAgentView.model_validate(payload)


def _validate_facts(
    target_view: CorpusTargetViewManifest,
    facts: NormalizedHardwareFacts,
) -> None:
    expected = (
        target_view.capacity_class_bytes,
        target_view.target.max_tensor_bytes,
        target_view.target.reference_ipc_limit_bytes,
    )
    observed = (
        facts.capacity_class_bytes,
        facts.max_tensor_bytes,
        facts.reference_ipc_limit_bytes,
    )
    if expected != observed:
        raise ValueError("Agent hardware facts differ from target view")
    if (
        facts.gfx_target is not None
        and facts.gfx_target != target_view.target.gfx_target.strip().lower()
    ):
        raise ValueError("Agent hardware identity differs from target view")
    if (
        facts.hardware_configuration_id is not None
        and facts.hardware_configuration_id
        != target_view.hardware_configuration_id
    ):
        raise ValueError(
            "Agent hardware configuration differs from target view"
        )


def _definition_view(
    entry: CorpusEntry,
    decision: GenerationDecision,
    records: list[GeneratedWorkloadRecord],
) -> AgentDefinitionView:
    if decision.status is not GenerationDecisionStatus.GENERATED:
        raise ValueError("generated workloads require a generated decision")
    if decision.distribution_id is None:
        raise ValueError("generated decision requires a distribution identity")
    development = tuple(
        sorted(
            (item for item in records if item.role is WorkloadRole.DEVELOPMENT),
            key=lambda item: item.slot_id,
        )
    )
    withheld = tuple(
        sorted(
            item.slot_id
            for item in records
            if item.role is WorkloadRole.HOLDOUT
        )
    )
    return AgentDefinitionView(
        semantic_id=entry.semantic_id,
        problem_name=entry.problem_name,
        definition_path=entry.definition_path,
        generation_rule_path=entry.generation_rule_path,
        generation_rule_sha256=entry.generation_rule_sha256,
        operation_family=entry.operation_family,
        distribution_id=decision.distribution_id,
        development_workloads=development,
        withheld_slot_ids=withheld,
    )


def _json_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_payload(item) for item in value]
    return value


__all__ = [
    "build_agent_view",
    "classify_hardware_shift",
    "hardware_facts",
]
