"""Synthetic metadata fixtures: no real target generation or input exposure."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.dataset.corpus import (
    load_corpus_manifest,
    load_target_descriptor,
    resolve_generation_context,
    semantic_fingerprint,
    workload_view_digest,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusProfile,
    CorpusTargetViewManifest,
    GeneratedProblem,
    GenerationDecision,
    GenerationDecisionStatus,
    TargetCoverageStatus,
    WorkloadGenerationRule,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.dataset.workload_generation import (
    _record,
    _workload,
    workload_requirements,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

ROOT = Path(__file__).resolve().parents[4]
RELEASE = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V2"


def _synthetic_source(root):
    """Use the schema vocabulary with toy source IDs, facts, rules and axes."""
    raw = yaml.safe_load((RELEASE / "manifest.yaml").read_text())
    entry = raw["entries"][0]
    definition = Definition.model_validate_json(
        (RELEASE / entry["definition_path"]).read_text()
    )
    rule = yaml.safe_load((RELEASE / entry["generation_rule_path"]).read_text())
    entry["semantic_id"] = rule["semantic_id"] = "synthetic.reader.matmul"
    entry["definition_path"] = "problem/definition.json"
    entry["generation_rule_path"] = "problem/rule.yaml"
    source_id = entry["source_ids"][0]
    raw["sources"] = [
        item for item in raw["sources"] if item["source_id"] == source_id
    ]
    raw["sources"][0]["architecture_facts"]["hidden_size"] = 64
    raw["sources"][0]["architecture_facts"]["intermediate_size"] = 64
    for index, slot in enumerate(rule["slots"], 1):
        slot.update(scale_numerator=index, scale_denominator=1, irregular=False)
    (root / "problem").mkdir(parents=True)
    atomic_write_json_value(
        root / entry["definition_path"],
        definition.model_dump(mode="json"),
        sort_keys=False,
    )
    (root / entry["generation_rule_path"]).write_text(yaml.safe_dump(rule))
    entry["definition_sha256"] = sha256_file(root / entry["definition_path"])
    entry["generation_rule_sha256"] = sha256_file(
        root / entry["generation_rule_path"]
    )
    entry["semantic_fingerprint"] = semantic_fingerprint(definition, entry)
    raw["entries"] = [entry]
    raw["corpus_id"] = "synthetic_reader_fixture"
    raw["coverage_policy"]["definition_count"] = 1
    raw["coverage_policy"]["operation_definition_counts"] = {
        key: int(key == "linear")
        for key in raw["coverage_policy"]["operation_definition_counts"]
    }
    raw["coverage_policy"]["profile_minimum_generated_definitions"] = {
        "core": 1
    }
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(raw))
    return (
        manifest_path,
        definition,
        WorkloadGenerationRule.model_validate(rule),
    )


def _synthetic_capacity():
    payload = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "synthetic CPU fixture, not observed hardware",
        "gfx_target": "gfx1200",
        "torch_version": "synthetic",
        "hip_version": "synthetic",
        "collected_at": datetime(2026, 1, 1, tzinfo=UTC),
        "runtime_free_bytes": 8 * 1024**3,
        "runtime_total_bytes": 8 * 1024**3,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": 8 * 1024**3,
        "harness_reserve_bytes": 0,
        "safety_percent": 50,
        "usable_budget_bytes": 4 * 1024**3,
        "capacity_probe_digest": "0" * 64,
    }
    payload["capacity_probe_digest"] = capacity_probe_digest(
        GPUMemoryQuotaEvidence.model_construct(
            _fields_set=set(payload), **payload
        )
    )
    return GPUMemoryQuotaEvidence.model_validate(payload)


@pytest.fixture
def synthetic_target_files(tmp_path):
    """Hand-authored toy metadata; never calls generate_corpus or generate_rule_workloads."""
    source_path, definition, rule = _synthetic_source(tmp_path / "source")
    source = load_corpus_manifest(source_path)
    target = load_target_descriptor(
        ROOT / "problems/LLM_CORE/targets/isa/gfx1200.yaml"
    )
    capacity = _synthetic_capacity()
    context = resolve_generation_context(
        source_path, source, target, (CorpusProfile.CORE,), capacity
    )
    workloads, records = [], []
    for index, slot in enumerate(rule.slots, 1):
        # These nine tiny, explicit toy shapes are unrelated to LLM_CORE target sizes.
        axes = {"M": index * 64, "K": 64, "N": 64}
        workload = _workload(
            definition,
            rule,
            slot,
            axes,
            context.cohort_id,
            context.entry_distributions[rule.semantic_id],
        )
        resources = workload_requirements(
            definition, axes, rule.operation_family, None, ()
        )
        workloads.append(workload)
        records.append(_record(rule, slot, workload, resources, 64))
    output = tmp_path / "view"
    (output / "problem").mkdir(parents=True)
    atomic_write_json_value(
        output / "problem/definition.json",
        definition.model_dump(mode="json"),
        sort_keys=False,
    )
    atomic_write_jsonl_values(output / "problem/workload.jsonl", workloads)
    problem = GeneratedProblem(
        semantic_id=rule.semantic_id,
        problem_name=source.entries[0].problem_name,
        definition_path="problem/definition.json",
        workload_path="problem/workload.jsonl",
        workload_uuids=tuple(w.uuid for w in workloads),
        definition_sha256=sha256_file(output / "problem/definition.json"),
        workload_sha256=sha256_file(output / "problem/workload.jsonl"),
    )
    decision = GenerationDecision(
        semantic_id=rule.semantic_id,
        status=GenerationDecisionStatus.GENERATED,
        detail="synthetic metadata fixture",
        common_scale=64,
        distribution_id=context.entry_distributions[rule.semantic_id],
        workload_uuids=problem.workload_uuids,
    )
    coverage = {
        "definitions": 1,
        "workloads": 9,
        "operation:linear": 1,
        **{f"profile:{p}": 1 for p in source.entries[0].profiles},
    }
    view = CorpusTargetViewManifest(
        schema_version=DatasetArtifactSchema.CORPUS_TARGET_VIEW,
        corpus_id=source.corpus_id,
        release_id=source.release_id,
        source_manifest_sha256=sha256_file(source_path),
        target_descriptor_sha256=stable_json_checksum(
            target.model_dump(mode="json")
        ),
        target=target,
        capacity_evidence=capacity,
        hardware_context=context.hardware,
        hardware_configuration_id=context.hardware.hardware_configuration_id,
        capacity_class_id=context.capacity_id,
        capacity_class_bytes=context.capacity_bytes,
        distribution_id=context.distribution_id,
        generation_cohort_id=context.cohort_id,
        generator_version="llm_core_common_scale.v2",
        workload_view_digest=workload_view_digest(
            source_path, source, context, tuple(records), (decision,)
        ),
        requested_profiles=(CorpusProfile.CORE,),
        require_complete_profile=True,
        coverage_status=TargetCoverageStatus.COMPLETE,
        coverage=coverage,
        problems=(problem,),
        workloads=tuple(records),
        decisions=(decision,),
    )
    view_path = output / "target-view-manifest.yaml"
    view_path.write_text(yaml.safe_dump(view.model_dump(mode="json")))
    return source_path, view_path
