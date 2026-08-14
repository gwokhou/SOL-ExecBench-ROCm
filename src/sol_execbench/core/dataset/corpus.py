# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Load, validate, and statically select generic problem corpora."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_jsonl_values,
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusProfile,
    CorpusSelectionManifest,
    CorpusWorkloadRecord,
    SelectedProblem,
    SelectionDecision,
    SelectionReason,
    StaticTargetDescriptor,
    TargetQualificationStatus,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    validate_relative_artifact_path,
)

SELECTION_MANIFEST_FILENAME = "selection-manifest.yaml"


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
    """Load a manifest and verify all committed problem artifacts."""
    manifest_path = Path(path).resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = CorpusManifest.model_validate(raw)
    _validate_artifacts(manifest_path.parent, manifest)
    _validate_coverage(manifest)
    return manifest


def load_target_descriptor(path: str | Path) -> StaticTargetDescriptor:
    """Load one JSON or YAML target descriptor."""
    descriptor_path = Path(path)
    raw = yaml.safe_load(descriptor_path.read_text(encoding="utf-8")) or {}
    descriptor = StaticTargetDescriptor.model_validate(raw)
    _require_declared_target(descriptor)
    return descriptor


def validate_corpus(path: str | Path) -> dict[str, Any]:
    """Return a compact validation and coverage report."""
    manifest = load_corpus_manifest(path)
    families = Counter(
        entry.operation_family.value for entry in manifest.entries
    )
    profiles = Counter(
        profile.value
        for entry in manifest.entries
        for workload in entry.workloads
        for profile in entry.profiles
    )
    return {
        "status": "valid",
        "corpus_id": manifest.corpus_id,
        "release_id": manifest.release_id,
        "release_state": manifest.release_state.value,
        "definitions": len(manifest.entries),
        "workloads": sum(len(entry.workloads) for entry in manifest.entries),
        "operation_families": dict(sorted(families.items())),
        "profile_workloads": dict(sorted(profiles.items())),
        "sources": len(manifest.sources),
    }


def select_corpus(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    target: StaticTargetDescriptor,
    profiles: tuple[CorpusProfile, ...],
    require_complete_profile: bool = False,
) -> Path:
    """Atomically emit a target-specific view using only static metadata."""
    _require_declared_target(target)
    if target.memory_budget_bytes is None:
        raise ValueError("target memory_budget_bytes must be declared")
    if not profiles:
        raise ValueError("at least one corpus profile must be selected")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_corpus_manifest(manifest_path)
    unknown = set(profiles) - set(manifest.profiles)
    if unknown:
        raise ValueError(f"unknown corpus profiles: {sorted(unknown)}")
    decisions = _select_workloads(manifest, target, frozenset(profiles))
    coverage = _selection_coverage(manifest, decisions)
    complete = _profile_coverage_complete(manifest, profiles, coverage)
    if require_complete_profile and not complete:
        raise ValueError("static selection does not satisfy profile coverage")
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"selection output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        problems = _write_selection(
            manifest_path.parent,
            staging,
            manifest,
            decisions,
        )
        record = _selection_manifest(
            manifest_path,
            manifest,
            target,
            profiles,
            require_complete_profile,
            complete,
            coverage,
            problems,
            decisions,
        )
        _write_yaml(staging / SELECTION_MANIFEST_FILENAME, record)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _require_declared_target(target: StaticTargetDescriptor) -> None:
    if target.qualification_status is not TargetQualificationStatus.DECLARED:
        raise ValueError(
            "hardware-qualified targets are deferred; use a declared descriptor",
        )


def _validate_artifacts(root: Path, manifest: CorpusManifest) -> None:
    for entry in manifest.entries:
        definition_path = _artifact_path(root, entry.definition_path)
        workload_path = _artifact_path(root, entry.workload_path)
        if sha256_file(definition_path) != entry.definition_sha256:
            raise ValueError(
                f"definition checksum mismatch: {entry.semantic_id}"
            )
        if sha256_file(workload_path) != entry.workload_sha256:
            raise ValueError(f"workload checksum mismatch: {entry.semantic_id}")
        definition = load_json_file(Definition, definition_path)
        workloads = load_jsonl_file(Workload, workload_path)
        validate_problem_contract(definition, workloads)
        expected = [item.uuid for item in entry.workloads]
        observed = [item.uuid for item in workloads]
        if observed != expected:
            raise ValueError(
                f"workload inventory mismatch: {entry.semantic_id}"
            )
        if (
            semantic_fingerprint(definition, entry)
            != entry.semantic_fingerprint
        ):
            raise ValueError(
                f"semantic fingerprint mismatch: {entry.semantic_id}"
            )
        tiers = {item.shape_tier for item in entry.workloads}
        if len(tiers) < 3:
            raise ValueError(
                f"fewer than three shape tiers: {entry.semantic_id}"
            )


def _artifact_path(root: Path, relative: str) -> Path:
    safe = validate_relative_artifact_path(relative)
    path = root / safe
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular corpus artifact: {safe}")
    if root != path.resolve().parent and root not in path.resolve().parents:
        raise ValueError(f"corpus artifact escapes manifest root: {safe}")
    return path


def _validate_coverage(manifest: CorpusManifest) -> None:
    policy = manifest.coverage_policy
    definition_count = len(manifest.entries)
    workload_count = sum(len(entry.workloads) for entry in manifest.entries)
    if definition_count < policy.minimum_definitions:
        raise ValueError(
            "corpus definition coverage is below its release floor"
        )
    if workload_count < policy.minimum_workloads:
        raise ValueError("corpus workload coverage is below its release floor")
    observed = Counter(entry.operation_family for entry in manifest.entries)
    for family, minimum in policy.operation_minimum_definitions.items():
        if observed[family] < minimum:
            raise ValueError(f"operation coverage below floor: {family.value}")


def _select_workloads(
    manifest: CorpusManifest,
    target: StaticTargetDescriptor,
    profiles: frozenset[CorpusProfile],
) -> tuple[SelectionDecision, ...]:
    decisions: list[SelectionDecision] = []
    for entry in manifest.entries:
        for workload in entry.workloads:
            reason, detail = static_selection_reason(
                entry,
                workload,
                target,
                profiles,
            )
            decisions.append(
                SelectionDecision(
                    semantic_id=entry.semantic_id,
                    workload_uuid=workload.uuid,
                    included=reason is SelectionReason.INCLUDED,
                    reason=reason,
                    detail=detail,
                ),
            )
    return tuple(decisions)


def static_selection_reason(
    entry: CorpusEntry,
    workload: CorpusWorkloadRecord,
    target: StaticTargetDescriptor,
    profiles: frozenset[CorpusProfile],
) -> tuple[SelectionReason, str]:
    """Evaluate one workload in the documented fail-closed stage order."""
    if target.memory_budget_bytes is None:
        raise ValueError("target memory_budget_bytes must be declared")
    if not profiles.intersection(entry.profiles):
        return (
            SelectionReason.PROFILE_NOT_SELECTED,
            "entry is outside requested profiles",
        )
    requirements = workload.requirements
    missing_dtypes = set(requirements.dtypes) - set(target.supported_dtypes)
    if missing_dtypes:
        return SelectionReason.UNSUPPORTED_DTYPE, _enum_detail(missing_dtypes)
    missing_quant = set(requirements.quantization) - set(
        target.supported_quantization,
    )
    if missing_quant:
        return SelectionReason.UNSUPPORTED_QUANTIZATION, _enum_detail(
            missing_quant
        )
    missing_capabilities = set(requirements.capabilities) - set(
        target.capabilities
    )
    if missing_capabilities:
        return SelectionReason.MISSING_CAPABILITY, _enum_detail(
            missing_capabilities
        )
    resources = requirements.resources
    if resources.max_tensor_bytes > target.max_tensor_bytes:
        return (
            SelectionReason.TENSOR_LIMIT_EXCEEDED,
            "maximum tensor exceeds target limit",
        )
    if resources.reference_ipc_bytes > target.reference_ipc_limit_bytes:
        return (
            SelectionReason.REFERENCE_IPC_LIMIT_EXCEEDED,
            "reference IPC storage exceeds target limit",
        )
    if resources.reference_peak_bytes > target.memory_budget_bytes:
        return (
            SelectionReason.MEMORY_BUDGET_EXCEEDED,
            "reference peak exceeds memory budget",
        )
    return SelectionReason.INCLUDED, "all static requirements satisfied"


def _enum_detail(values: set[Any]) -> str:
    return "missing: " + ", ".join(sorted(str(value) for value in values))


def _selection_coverage(
    manifest: CorpusManifest,
    decisions: tuple[SelectionDecision, ...],
) -> dict[str, int]:
    included = {
        (item.semantic_id, item.workload_uuid)
        for item in decisions
        if item.included
    }
    coverage: Counter[str] = Counter()
    selected_definitions: set[str] = set()
    for entry in manifest.entries:
        for workload in entry.workloads:
            if (entry.semantic_id, workload.uuid) not in included:
                continue
            selected_definitions.add(entry.semantic_id)
            coverage[f"operation:{entry.operation_family.value}"] += 1
            for profile in entry.profiles:
                coverage[f"profile:{profile.value}"] += 1
    coverage["definitions"] = len(selected_definitions)
    coverage["workloads"] = len(included)
    return dict(sorted(coverage.items()))


def _profile_coverage_complete(
    manifest: CorpusManifest,
    profiles: tuple[CorpusProfile, ...],
    coverage: dict[str, int],
) -> bool:
    floors = manifest.coverage_policy.profile_minimum_workloads
    return all(
        coverage.get(f"profile:{profile.value}", 0) >= floors.get(profile, 0)
        for profile in profiles
    )


def _write_selection(
    source_root: Path,
    staging: Path,
    manifest: CorpusManifest,
    decisions: tuple[SelectionDecision, ...],
) -> tuple[SelectedProblem, ...]:
    included = {
        (item.semantic_id, item.workload_uuid)
        for item in decisions
        if item.included
    }
    selected: list[SelectedProblem] = []
    for entry in manifest.entries:
        workloads = load_jsonl_file(Workload, source_root / entry.workload_path)
        workloads = [
            item
            for item in workloads
            if (entry.semantic_id, item.uuid) in included
        ]
        if not workloads:
            continue
        definition_target = staging / entry.definition_path
        workload_target = staging / entry.workload_path
        definition_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / entry.definition_path, definition_target)
        atomic_write_jsonl_values(workload_target, workloads)
        selected.append(
            SelectedProblem(
                semantic_id=entry.semantic_id,
                problem_name=entry.problem_name,
                definition_path=entry.definition_path,
                workload_path=entry.workload_path,
                workload_uuids=tuple(item.uuid for item in workloads),
                definition_sha256=sha256_file(definition_target),
                workload_sha256=sha256_file(workload_target),
            ),
        )
    return tuple(selected)


def _selection_manifest(
    manifest_path: Path,
    manifest: CorpusManifest,
    target: StaticTargetDescriptor,
    profiles: tuple[CorpusProfile, ...],
    strict: bool,
    complete: bool,
    coverage: dict[str, int],
    problems: tuple[SelectedProblem, ...],
    decisions: tuple[SelectionDecision, ...],
) -> CorpusSelectionManifest:
    return CorpusSelectionManifest(
        schema_version=DatasetArtifactSchema.CORPUS_SELECTION_MANIFEST,
        corpus_id=manifest.corpus_id,
        release_id=manifest.release_id,
        source_manifest_sha256=sha256_file(manifest_path),
        target_descriptor_sha256=stable_json_checksum(
            target.model_dump(mode="json"),
        ),
        target=target,
        requested_profiles=profiles,
        require_complete_profile=strict,
        coverage_complete=complete,
        coverage=coverage,
        problems=problems,
        decisions=decisions,
    )


def _write_yaml(path: Path, model: CorpusSelectionManifest) -> None:
    path.write_text(
        yaml.safe_dump(
            json.loads(model.model_dump_json()),
            sort_keys=False,
        ),
        encoding="utf-8",
    )


__all__ = [
    "SELECTION_MANIFEST_FILENAME",
    "load_corpus_manifest",
    "load_target_descriptor",
    "select_corpus",
    "semantic_fingerprint",
    "static_selection_reason",
    "validate_corpus",
]
