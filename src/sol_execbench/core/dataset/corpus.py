# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Load rule-defined corpora and generate measured target views."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import (
    atomic_write_jsonl_values,
    load_json_file,
)
from sol_execbench.core.dataset.corpus_models import (
    WORKLOAD_GENERATION_PROTOCOL_MAJOR,
    WORKLOAD_GENERATOR_VERSION,
    CorpusEntry,
    CorpusManifest,
    CorpusProfile,
    CorpusTargetViewManifest,
    GeneratedProblem,
    GeneratedWorkloadRecord,
    GenerationDecision,
    GenerationDecisionStatus,
    QuantizationScheme,
    StaticCapability,
    StaticTargetDescriptor,
    TargetCoverageStatus,
    TargetQualificationStatus,
    WorkloadGenerationRule,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.dataset.workload_generation import (
    GeneratedRuleResult,
    capacity_class_bytes,
    distribution_id,
    generate_rule_workloads,
    generation_cohort_id,
)
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    validate_relative_artifact_path,
)
from sol_execbench.core.platform.memory_quota import GPUMemoryQuotaEvidence

TARGET_VIEW_MANIFEST_FILENAME = "target-view-manifest.yaml"


@dataclass(frozen=True, slots=True, kw_only=True)
class _GenerationContext:
    capacity_bytes: int
    capacity_id: str | None
    entry_distributions: dict[str, str]
    distribution_id: str
    cohort_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _GenerationState:
    results: dict[str, GeneratedRuleResult]
    decisions: tuple[GenerationDecision, ...]
    coverage: dict[str, int]
    complete: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class _ViewRequest:
    manifest_path: Path
    manifest: CorpusManifest
    target: StaticTargetDescriptor
    profiles: tuple[CorpusProfile, ...]
    capacity: GPUMemoryQuotaEvidence
    context: _GenerationContext
    state: _GenerationState
    strict: bool


def semantic_fingerprint(
    definition: Definition,
    entry: CorpusEntry | dict[str, Any],
) -> str:
    """Return the source- and shape-independent semantic fingerprint."""
    family = (
        entry.operation_family
        if isinstance(entry, CorpusEntry)
        else entry["operation_family"]
    )
    payload = {
        "operation_family": str(family),
        "op_type": definition.op_type,
        "inputs": {
            name: {"shape": spec.shape, "dtype": spec.dtype}
            for name, spec in definition.inputs.items()
        },
        "outputs": {
            name: {"shape": spec.shape, "dtype": spec.dtype}
            for name, spec in definition.outputs.items()
        },
        "reference": definition.reference,
    }
    return stable_json_checksum(payload)


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    """Load a manifest and validate every Definition and generation rule."""
    manifest_path = Path(path).resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = CorpusManifest.model_validate(raw)
    _validate_artifacts(manifest_path.parent, manifest)
    return manifest


def load_target_descriptor(path: str | Path) -> StaticTargetDescriptor:
    """Load one declared target descriptor."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    descriptor = StaticTargetDescriptor.model_validate(raw)
    if (
        descriptor.qualification_status
        is not TargetQualificationStatus.DECLARED
    ):
        raise ValueError("target descriptor must have declared status")
    return descriptor


def validate_corpus(path: str | Path) -> dict[str, Any]:
    """Return a compact validation report for a rule-defined corpus."""
    manifest = load_corpus_manifest(path)
    families = Counter(
        entry.operation_family.value for entry in manifest.entries
    )
    return {
        "status": "valid",
        "corpus_id": manifest.corpus_id,
        "release_id": manifest.release_id,
        "release_state": manifest.release_state.value,
        "definitions": len(manifest.entries),
        "generation_rules": len(manifest.entries),
        "concrete_workloads": 0,
        "operation_families": dict(sorted(families.items())),
        "sources": len(manifest.sources),
    }


def generate_corpus(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    target: StaticTargetDescriptor,
    profiles: tuple[CorpusProfile, ...],
    capacity_evidence: GPUMemoryQuotaEvidence,
    require_complete_profile: bool = False,
) -> Path:
    """Atomically generate concrete workloads from rules and measured capacity."""
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"generation output already exists: {output}")
    _validate_generation_request(target, capacity_evidence, profiles)
    manifest_path = Path(manifest_path).resolve()
    manifest = load_corpus_manifest(manifest_path)
    unknown = set(profiles) - set(manifest.profiles)
    if unknown:
        raise ValueError(f"unknown corpus profiles: {sorted(unknown)}")
    context = _generation_context(
        manifest_path,
        manifest,
        target,
        profiles,
        capacity_evidence,
    )
    results, decisions = _generate_entries(
        manifest_path.parent,
        manifest,
        target,
        profiles,
        context,
    )
    coverage = _generation_coverage(manifest, results)
    complete = _coverage_complete(manifest, profiles, coverage)
    if require_complete_profile and not complete:
        raise ValueError(
            "generated target view does not satisfy profile coverage"
        )
    request = _ViewRequest(
        manifest_path=manifest_path,
        manifest=manifest,
        target=target,
        profiles=profiles,
        capacity=capacity_evidence,
        context=context,
        state=_GenerationState(
            results=results,
            decisions=decisions,
            coverage=coverage,
            complete=complete,
        ),
        strict=require_complete_profile,
    )
    return _write_generated_view(output, request)


def _validate_generation_request(
    target: StaticTargetDescriptor,
    capacity: GPUMemoryQuotaEvidence,
    profiles: tuple[CorpusProfile, ...],
) -> None:
    if target.qualification_status is not TargetQualificationStatus.DECLARED:
        raise ValueError("target descriptor must have declared status")
    if capacity.gfx_target.strip().lower() != target.gfx_target.strip().lower():
        raise ValueError(
            "capacity evidence gfx target does not match descriptor"
        )
    if not profiles:
        raise ValueError("at least one corpus profile must be selected")


def _validate_artifacts(root: Path, manifest: CorpusManifest) -> None:
    for entry in manifest.entries:
        definition_path = _artifact_path(root, entry.definition_path)
        rule_path = _artifact_path(root, entry.generation_rule_path)
        if sha256_file(definition_path) != entry.definition_sha256:
            raise ValueError(
                f"definition checksum mismatch: {entry.semantic_id}"
            )
        if sha256_file(rule_path) != entry.generation_rule_sha256:
            raise ValueError(
                f"generation rule checksum mismatch: {entry.semantic_id}"
            )
        definition = load_json_file(Definition, definition_path)
        rule = _load_rule(rule_path)
        if rule.semantic_id != entry.semantic_id:
            raise ValueError(
                f"generation rule identity mismatch: {entry.semantic_id}"
            )
        if rule.operation_family is not entry.operation_family:
            raise ValueError(
                f"generation rule family mismatch: {entry.semantic_id}"
            )
        if (
            semantic_fingerprint(definition, entry)
            != entry.semantic_fingerprint
        ):
            raise ValueError(
                f"semantic fingerprint mismatch: {entry.semantic_id}"
            )


def _artifact_path(root: Path, relative: str) -> Path:
    safe = validate_relative_artifact_path(relative)
    path = root / safe
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular corpus artifact: {safe}")
    if root != path.resolve().parent and root not in path.resolve().parents:
        raise ValueError(f"corpus artifact escapes manifest root: {safe}")
    return path


def _load_rule(path: Path) -> WorkloadGenerationRule:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return WorkloadGenerationRule.model_validate(raw)


def _generation_context(
    manifest_path: Path,
    manifest: CorpusManifest,
    target: StaticTargetDescriptor,
    profiles: tuple[CorpusProfile, ...],
    capacity: GPUMemoryQuotaEvidence,
) -> _GenerationContext:
    capacity_bytes = capacity_class_bytes(
        capacity.usable_budget_bytes,
        manifest.generation_policy.capacity_classes_gib,
    )
    entry_distributions = _entry_distribution_ids(
        manifest_path.parent, manifest
    )
    overall_distribution = stable_json_checksum(
        {
            "profiles": sorted(profile.value for profile in profiles),
            "entry_distributions": entry_distributions,
        }
    )
    cohort = generation_cohort_id(
        distribution_id=overall_distribution,
        target=target,
        capacity_class_bytes=capacity_bytes,
        capacity_numerator=(
            manifest.generation_policy.maximum_capacity_numerator
        ),
        capacity_denominator=(
            manifest.generation_policy.maximum_capacity_denominator
        ),
        executor_version=WORKLOAD_GENERATOR_VERSION,
    )
    return _GenerationContext(
        capacity_bytes=capacity_bytes,
        capacity_id=(
            f"mem-{capacity_bytes // (1024**3)}gib" if capacity_bytes else None
        ),
        entry_distributions=entry_distributions,
        distribution_id=overall_distribution,
        cohort_id=cohort,
    )


def _entry_distribution_ids(
    root: Path,
    manifest: CorpusManifest,
) -> dict[str, str]:
    facts = {
        source.source_id: source.architecture_facts
        for source in manifest.sources
    }
    identities: dict[str, str] = {}
    for entry in manifest.entries:
        definition = load_json_file(Definition, root / entry.definition_path)
        rule = _load_rule(root / entry.generation_rule_path)
        identities[entry.semantic_id] = distribution_id(
            semantic_fingerprint=entry.semantic_fingerprint,
            definition=definition,
            rule=rule,
            facts=facts[entry.source_ids[0]],
        )
    return dict(sorted(identities.items()))


def _generate_entries(
    root: Path,
    manifest: CorpusManifest,
    target: StaticTargetDescriptor,
    profiles: tuple[CorpusProfile, ...],
    context: _GenerationContext,
) -> tuple[dict[str, GeneratedRuleResult], tuple[GenerationDecision, ...]]:
    facts = {
        source.source_id: source.architecture_facts
        for source in manifest.sources
    }
    results: dict[str, GeneratedRuleResult] = {}
    decisions: list[GenerationDecision] = []
    selected_profiles = frozenset(profiles)
    for entry in manifest.entries:
        definition = load_json_file(Definition, root / entry.definition_path)
        rule = _load_rule(root / entry.generation_rule_path)
        status, detail = _eligibility(
            entry, definition, rule, target, selected_profiles
        )
        result = None
        if (
            status is GenerationDecisionStatus.GENERATED
            and context.capacity_bytes
        ):
            result = generate_rule_workloads(
                definition=definition,
                rule=rule,
                facts=facts[entry.source_ids[0]],
                target=target,
                capacity_bytes=context.capacity_bytes,
                cohort_id=context.cohort_id,
                semantic_fingerprint=entry.semantic_fingerprint,
            )
            if result is None:
                status = GenerationDecisionStatus.INSUFFICIENT_CAPACITY
                detail = "no common scale preserves all nine distinct slots"
        elif status is GenerationDecisionStatus.GENERATED:
            status = GenerationDecisionStatus.INSUFFICIENT_CAPACITY
            detail = "usable quota is below the 1 GiB capacity class"
        if result is not None:
            results[entry.semantic_id] = result
        decisions.append(_decision(entry, context, status, detail, result))
    return results, tuple(decisions)


def _eligibility(
    entry: CorpusEntry,
    definition: Definition,
    rule: WorkloadGenerationRule,
    target: StaticTargetDescriptor,
    profiles: frozenset[CorpusProfile],
) -> tuple[GenerationDecisionStatus, str]:
    if not profiles.intersection(entry.profiles):
        return (
            GenerationDecisionStatus.PROFILE_NOT_SELECTED,
            "profile not requested",
        )
    normalized_gfx = target.gfx_target.strip().lower()
    eligible_gfx = {
        value.strip().lower() for value in rule.eligible_gfx_targets
    }
    if normalized_gfx not in eligible_gfx:
        return (
            GenerationDecisionStatus.TARGET_NOT_ELIGIBLE,
            "gfx target not eligible",
        )
    required_dtypes = _rule_dtypes(definition, rule.quantization)
    if missing := required_dtypes - set(target.supported_dtypes):
        return GenerationDecisionStatus.UNSUPPORTED_DTYPE, _missing_detail(
            missing
        )
    if (
        rule.quantization
        and rule.quantization not in target.supported_quantization
    ):
        return GenerationDecisionStatus.UNSUPPORTED_QUANTIZATION, str(
            rule.quantization
        )
    required_caps = {StaticCapability.DENSE_TENSOR, *rule.capabilities}
    if missing := required_caps - set(target.capabilities):
        return GenerationDecisionStatus.MISSING_CAPABILITY, _missing_detail(
            missing
        )
    return GenerationDecisionStatus.GENERATED, "eligible rule"


def _rule_dtypes(
    definition: Definition,
    quantization: QuantizationScheme | None,
) -> set[DType]:
    dtypes = {
        tensor.dtype
        for tensor in (
            *definition.inputs.values(),
            *definition.outputs.values(),
        )
    }
    extra = {
        QuantizationScheme.FP8_PER_TENSOR: DType.FLOAT8_E4M3FN,
        QuantizationScheme.FP8_PER_TOKEN: DType.FLOAT8_E4M3FN,
        QuantizationScheme.MXFP8_BLOCK: DType.FLOAT8_E5M2,
        QuantizationScheme.FP4_GROUP: DType.FLOAT4_E2M1FN_X2,
        QuantizationScheme.MXFP4_BLOCK: DType.FLOAT4_E2M1FN_X2,
        QuantizationScheme.INT8_PER_TENSOR: DType.INT8,
        QuantizationScheme.INT8_PER_TOKEN: DType.INT8,
        QuantizationScheme.INT8_WEIGHT_ONLY: DType.INT8,
    }
    if quantization is not None:
        dtypes.add(extra[quantization])
    return dtypes


def _missing_detail(values: set[Any]) -> str:
    return "missing: " + ", ".join(sorted(map(str, values)))


def _decision(
    entry: CorpusEntry,
    context: _GenerationContext,
    status: GenerationDecisionStatus,
    detail: str,
    result: GeneratedRuleResult | None,
) -> GenerationDecision:
    return GenerationDecision(
        semantic_id=entry.semantic_id,
        status=status,
        detail=detail,
        distribution_id=context.entry_distributions[entry.semantic_id],
        common_scale=None if result is None else result.common_scale,
        workload_uuids=(
            ()
            if result is None
            else tuple(item.uuid for item in result.workloads)
        ),
    )


def _generation_coverage(
    manifest: CorpusManifest,
    results: dict[str, GeneratedRuleResult],
) -> dict[str, int]:
    coverage: Counter[str] = Counter()
    for entry in manifest.entries:
        result = results.get(entry.semantic_id)
        if result is None:
            continue
        coverage["definitions"] += 1
        coverage["workloads"] += len(result.workloads)
        coverage[f"operation:{entry.operation_family.value}"] += 1
        for profile in entry.profiles:
            coverage[f"profile:{profile.value}"] += 1
    return dict(sorted(coverage.items()))


def _coverage_complete(
    manifest: CorpusManifest,
    profiles: tuple[CorpusProfile, ...],
    coverage: dict[str, int],
) -> bool:
    floors = manifest.coverage_policy.profile_minimum_generated_definitions
    return all(
        coverage.get(f"profile:{profile.value}", 0) >= floors.get(profile, 0)
        for profile in profiles
    )


def _write_generated_view(
    output_root: str | Path,
    request: _ViewRequest,
) -> Path:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"generation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        problems = _write_problems(
            request.manifest_path.parent,
            staging,
            request.manifest,
            request.state.results,
        )
        records = tuple(
            record
            for entry in request.manifest.entries
            for record in _records(request.state.results, entry)
        )
        view_digest = _workload_view_digest(
            request.manifest_path,
            request.manifest,
            request.context,
            records,
            request.state.decisions,
        )
        view = _target_view(request, problems, records, view_digest)
        _write_yaml(staging / TARGET_VIEW_MANIFEST_FILENAME, view)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _records(
    results: dict[str, GeneratedRuleResult],
    entry: CorpusEntry,
) -> tuple[GeneratedWorkloadRecord, ...]:
    result = results.get(entry.semantic_id)
    return () if result is None else result.records


def _write_problems(
    source_root: Path,
    staging: Path,
    manifest: CorpusManifest,
    results: dict[str, GeneratedRuleResult],
) -> tuple[GeneratedProblem, ...]:
    problems: list[GeneratedProblem] = []
    for entry in manifest.entries:
        result = results.get(entry.semantic_id)
        if result is None:
            continue
        definition_target = staging / entry.definition_path
        workload_relative = str(
            Path(entry.definition_path).with_name("workload.jsonl")
        )
        workload_target = staging / workload_relative
        definition_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / entry.definition_path, definition_target)
        atomic_write_jsonl_values(workload_target, list(result.workloads))
        problems.append(
            GeneratedProblem(
                semantic_id=entry.semantic_id,
                problem_name=entry.problem_name,
                definition_path=entry.definition_path,
                workload_path=workload_relative,
                workload_uuids=tuple(item.uuid for item in result.workloads),
                definition_sha256=sha256_file(definition_target),
                workload_sha256=sha256_file(workload_target),
            )
        )
    return tuple(problems)


def _workload_view_digest(
    manifest_path: Path,
    manifest: CorpusManifest,
    context: _GenerationContext,
    records: tuple[GeneratedWorkloadRecord, ...],
    decisions: tuple[GenerationDecision, ...],
) -> str:
    fingerprints = {
        entry.semantic_id: entry.semantic_fingerprint
        for entry in manifest.entries
    }
    workload_payload = [
        {
            "semantic_fingerprint": fingerprints[record.semantic_id],
            "slot_id": record.slot_id,
            "uuid": record.uuid,
            "axes": record.axes,
            "input_profile_id": record.input_profile_id,
            "input_profile_digest": record.input_profile_digest,
            "correctness_profile_id": record.correctness_profile_id,
            "correctness_profile_digest": record.correctness_profile_digest,
            "resource_envelope": record.requirements.resources.model_dump(
                mode="json"
            ),
        }
        for record in records
    ]
    return stable_json_checksum(
        {
            "corpus_manifest_digest": sha256_file(manifest_path),
            "generation_cohort_id": context.cohort_id,
            "generation_protocol_major": WORKLOAD_GENERATION_PROTOCOL_MAJOR,
            "generation_protocol_version": WORKLOAD_GENERATOR_VERSION,
            "workloads": workload_payload,
            "decisions": [
                {
                    "semantic_fingerprint": fingerprints[decision.semantic_id],
                    "status": decision.status,
                }
                for decision in decisions
            ],
        }
    )


def _target_view(
    request: _ViewRequest,
    problems: tuple[GeneratedProblem, ...],
    records: tuple[GeneratedWorkloadRecord, ...],
    view_digest: str,
) -> CorpusTargetViewManifest:
    status = (
        TargetCoverageStatus.COMPLETE
        if request.state.complete
        else TargetCoverageStatus.INSUFFICIENT_CAPACITY_COVERAGE
    )
    return CorpusTargetViewManifest(
        schema_version=DatasetArtifactSchema.CORPUS_TARGET_VIEW,
        corpus_id=request.manifest.corpus_id,
        release_id=request.manifest.release_id,
        source_manifest_sha256=sha256_file(request.manifest_path),
        target_descriptor_sha256=stable_json_checksum(
            request.target.model_dump(mode="json")
        ),
        target=request.target,
        capacity_evidence=request.capacity,
        capacity_class_id=request.context.capacity_id,
        capacity_class_bytes=request.context.capacity_bytes,
        distribution_id=request.context.distribution_id,
        generation_cohort_id=request.context.cohort_id,
        generator_version=WORKLOAD_GENERATOR_VERSION,
        workload_view_digest=view_digest,
        requested_profiles=request.profiles,
        require_complete_profile=request.strict,
        coverage_status=status,
        coverage=request.state.coverage,
        problems=problems,
        workloads=records,
        decisions=request.state.decisions,
    )


def _write_yaml(path: Path, model: CorpusTargetViewManifest) -> None:
    path.write_text(
        yaml.safe_dump(json.loads(model.model_dump_json()), sort_keys=False),
        encoding="utf-8",
    )


__all__ = [
    "TARGET_VIEW_MANIFEST_FILENAME",
    "generate_corpus",
    "load_corpus_manifest",
    "load_target_descriptor",
    "semantic_fingerprint",
    "validate_corpus",
]
