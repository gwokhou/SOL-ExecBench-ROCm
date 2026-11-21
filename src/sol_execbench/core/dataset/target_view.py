"""Read and authenticate an existing target view without generating workloads."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.corpus import (
    load_corpus_manifest,
    resolve_generation_context,
    workload_view_digest,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusProfile,
    CorpusTargetViewManifest,
    GeneratedProblem,
    GeneratedWorkloadRecord,
    GenerationDecisionStatus,
    TargetCoverageStatus,
    WorkloadGenerationRule,
)
from sol_execbench.core.dataset.workload_generation import (
    validate_materialized_rule,
)
from sol_execbench.core.integrity import (
    sha256_file,
    validate_relative_artifact_path,
)

_MAX_VIEW_BYTES = 16 * 1024**2
_MAX_PROBLEM_BYTES = 2 * 1024**2


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializedTargetView:
    """Validated source and already materialized problem locations."""

    path: Path
    source: CorpusManifest
    view: CorpusTargetViewManifest
    definitions: dict[str, Definition]
    workloads: dict[str, Workload]
    problem_dirs: dict[str, Path]


def scoped_artifact(root: Path, relative: str) -> Path:
    """Require a regular file within the owning artifact directory."""
    path = root / validate_relative_artifact_path(relative)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular target artifact: {relative}")
    path.resolve().relative_to(root.resolve())
    return path


def _safe_yaml_load(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML artifact: {path}") from exc


def load_materialized_target_view(
    source_path: Path,
    view_path: Path,
) -> MaterializedTargetView:
    """Validate source, identities, decisions and each concrete workload."""
    source_path, view_path = source_path.resolve(), view_path.resolve()
    if (
        source_path.stat().st_size > _MAX_VIEW_BYTES
        or view_path.stat().st_size > _MAX_VIEW_BYTES
    ):
        raise ValueError("target/source metadata byte budget exceeded")
    source = load_corpus_manifest(source_path)
    view = CorpusTargetViewManifest.model_validate(_safe_yaml_load(view_path))
    _validate_identity(source_path, source, view)
    _validate_inventory(source, view)
    definitions: dict[str, Definition] = {}
    workloads: dict[str, Workload] = {}
    directories: dict[str, Path] = {}
    for problem in view.problems:
        definition, concrete, directory = _load_problem(
            view_path.parent, problem
        )
        entry = next(
            item
            for item in source.entries
            if item.semantic_id == problem.semantic_id
        )
        if (
            problem.definition_sha256 != entry.definition_sha256
            or problem.definition_path != entry.definition_path
            or problem.problem_name != entry.problem_name
        ):
            raise ValueError(
                "materialized definition differs from frozen source"
            )
        rule = WorkloadGenerationRule.model_validate(
            _safe_yaml_load(
                scoped_artifact(source_path.parent, entry.generation_rule_path)
            )
        )
        records = tuple(
            item
            for item in view.workloads
            if item.semantic_id == entry.semantic_id
        )
        _validate_entry_workloads(
            source, view, entry, definition, concrete, records, rule
        )
        definitions[entry.semantic_id] = definition
        directories[entry.semantic_id] = directory
        workloads.update(concrete)
    return MaterializedTargetView(
        path=view_path,
        source=source,
        view=view,
        definitions=definitions,
        workloads=workloads,
        problem_dirs=directories,
    )


def _validate_identity(
    source_path: Path,
    source: CorpusManifest,
    view: CorpusTargetViewManifest,
) -> None:
    if (
        view.source_manifest_sha256 != sha256_file(source_path)
        or view.corpus_id != source.corpus_id
        or view.release_id != source.release_id
    ):
        raise ValueError("target view source identity mismatch")
    context = resolve_generation_context(
        source_path,
        source,
        view.target,
        view.requested_profiles,
        view.capacity_evidence,
    )
    if (
        view.distribution_id != context.distribution_id
        or view.generation_cohort_id != context.cohort_id
        or view.hardware_context != context.hardware
        or view.capacity_class_bytes != context.capacity_bytes
        or view.capacity_class_id != context.capacity_id
    ):
        raise ValueError("target view generation or hardware identity mismatch")
    for decision in view.decisions:
        if decision.distribution_id != context.entry_distributions.get(
            decision.semantic_id
        ):
            raise ValueError("decision distribution identity mismatch")
    expected = workload_view_digest(
        source_path, source, context, view.workloads, view.decisions
    )
    if expected != view.workload_view_digest:
        raise ValueError("target view digest mismatch")


def _validate_inventory(
    source: CorpusManifest, view: CorpusTargetViewManifest
) -> None:
    if (
        not view.require_complete_profile
        or view.coverage_status is not TargetCoverageStatus.COMPLETE
    ):
        raise ValueError("partial target view cannot enter a complete audit")
    selected = {
        entry.semantic_id
        for entry in source.entries
        if set(entry.profiles).intersection(view.requested_profiles)
    }
    decisions = {item.semantic_id: item for item in view.decisions}
    if len(decisions) != len(view.decisions) or set(decisions) != {
        e.semantic_id for e in source.entries
    }:
        raise ValueError("decision inventory differs from source")
    ids = [item.semantic_id for item in view.problems]
    uuids = [item.uuid for item in view.workloads]
    if (
        len(ids) != len(set(ids))
        or set(ids) != selected
        or len(uuids) != len(set(uuids))
    ):
        raise ValueError("duplicate or incomplete target inventory")
    if len(uuids) != 9 * len(ids):
        raise ValueError("target inventory requires all nine slots")
    expected_coverage: dict[str, int] = {
        "definitions": len(ids),
        "workloads": len(uuids),
    }
    for entry in source.entries:
        decision = decisions[entry.semantic_id]
        expected = (
            GenerationDecisionStatus.GENERATED
            if entry.semantic_id in selected
            else GenerationDecisionStatus.PROFILE_NOT_SELECTED
        )
        if decision.status is not expected:
            raise ValueError(
                "target decision status differs from selected profile"
            )
        if expected is GenerationDecisionStatus.GENERATED:
            records = [
                r for r in view.workloads if r.semantic_id == entry.semantic_id
            ]
            if (
                set(decision.workload_uuids) != {r.uuid for r in records}
                or len(decision.workload_uuids) != len(records)
                or any(r.common_scale != decision.common_scale for r in records)
            ):
                raise ValueError("decision workload identity mismatch")
            keys = [
                f"operation:{entry.operation_family}",
                *(f"profile:{p}" for p in entry.profiles),
            ]
            for key in keys:
                expected_coverage[key] = expected_coverage.get(key, 0) + 1
        elif decision.workload_uuids or decision.common_scale is not None:
            raise ValueError("excluded decision carries generated workloads")
    if view.coverage != expected_coverage:
        raise ValueError("target coverage counters differ from inventory")
    for profile in view.requested_profiles:
        floor = (
            source.coverage_policy.profile_minimum_generated_definitions.get(
                profile, 0
            )
        )
        if expected_coverage.get(f"profile:{profile}", 0) < floor:
            raise ValueError("target profile floor not met")


def _load_problem(
    root: Path,
    problem: GeneratedProblem,
) -> tuple[Definition, dict[str, Workload], Path]:
    definition_path = scoped_artifact(root, problem.definition_path)
    workload_path = scoped_artifact(root, problem.workload_path)
    if (
        definition_path.name != "definition.json"
        or workload_path.name != "workload.jsonl"
        or definition_path.parent != workload_path.parent
    ):
        raise ValueError(
            "target artifacts must form a materialized problem directory"
        )
    if (
        sha256_file(definition_path) != problem.definition_sha256
        or sha256_file(workload_path) != problem.workload_sha256
    ):
        raise ValueError("materialized problem checksum mismatch")
    if (
        max(definition_path.stat().st_size, workload_path.stat().st_size)
        > _MAX_PROBLEM_BYTES
    ):
        raise ValueError("materialized problem byte budget exceeded")
    definition = load_json_file(Definition, definition_path)
    concrete = load_jsonl_file(Workload, workload_path)
    by_uuid = {item.uuid: item for item in concrete}
    if (
        len(by_uuid) != len(concrete)
        or len(set(problem.workload_uuids)) != 9
        or set(by_uuid) != set(problem.workload_uuids)
    ):
        raise ValueError("materialized workload UUID inventory mismatch")
    return definition, by_uuid, definition_path.parent


def _validate_entry_workloads(
    source: CorpusManifest,
    view: CorpusTargetViewManifest,
    entry: CorpusEntry,
    definition: Definition,
    concrete: dict[str, Workload],
    records: tuple[GeneratedWorkloadRecord, ...],
    rule: WorkloadGenerationRule,
) -> None:
    if {record.uuid for record in records} != set(concrete):
        raise ValueError(
            "target workload UUID membership differs from its problem"
        )
    decision = next(
        d for d in view.decisions if d.semantic_id == entry.semantic_id
    )
    facts = next(
        s.architecture_facts
        for s in source.sources
        if s.source_id == entry.source_ids[0]
    )
    if decision.distribution_id is None:
        raise ValueError("generated decision lacks distribution identity")
    validate_materialized_rule(
        definition,
        rule,
        facts,
        view.target,
        view.generation_cohort_id,
        decision.distribution_id,
        view.capacity_class_bytes
        * source.generation_policy.maximum_capacity_numerator
        // source.generation_policy.maximum_capacity_denominator,
        tuple(concrete[r.uuid] for r in records),
        records,
    )


def require_llm_core_scope(target: MaterializedTargetView) -> None:
    """Apply the frozen C0 denominator without generating a target view."""
    if (
        target.view.requested_profiles != (CorpusProfile.CORE,)
        or len(target.source.entries) != 36
        or len(target.view.problems) != 20
        or len(target.view.workloads) != 180
    ):
        raise ValueError(
            "M0-C0 requires 36 decisions, 20 core definitions and 180 workloads"
        )
