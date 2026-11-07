from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.bench.input_generation import gen_inputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset import workload_generation
from sol_execbench.core.dataset.corpus import (
    generate_corpus,
    load_corpus_manifest,
    load_target_descriptor,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import (
    WORKLOAD_GENERATION_PROTOCOL_MAJOR,
    CorpusManifest,
    CorpusOperationFamily,
    CorpusProfile,
    CorpusTargetViewManifest,
    GenerationDecisionStatus,
    StaticTargetDescriptor,
    WorkloadGenerationRule,
)
from sol_execbench.core.integrity import stable_json_checksum
from sol_execbench.core.platform.hardware import HardwareConfigurationKind
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
    derive_usable_budget,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

ROOT = Path(__file__).resolve().parents[4]
RELEASE = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V2"
MANIFEST = RELEASE / "manifest.yaml"
TARGETS = ROOT / "problems/LLM_CORE/targets"
GENERATOR = ROOT / "scripts/build_llm_core_corpus.py"
GIB = 1024**3


@pytest.fixture(scope="module")
def corpus() -> CorpusManifest:
    return load_corpus_manifest(MANIFEST)


def _capacity(
    free_gib: int,
    *,
    gfx_target: str = "gfx1200",
    collected_at: datetime | None = None,
    gpu_name: str = "mock GPU",
    device: str = "cuda:0",
    device_index: int = 0,
) -> GPUMemoryQuotaEvidence:
    free = free_gib * GIB
    usable = derive_usable_budget(
        runtime_free_bytes=free,
        environment_quota_bytes=None,
        stable_allocatable_bytes=free,
        harness_reserve_bytes=0,
    )
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": device,
        "device_index": device_index,
        "gpu_name": gpu_name,
        "gfx_target": gfx_target,
        "torch_version": "test",
        "hip_version": "test",
        "collected_at": collected_at or datetime(2026, 8, 15, tzinfo=UTC),
        "runtime_free_bytes": free,
        "runtime_total_bytes": free,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": free,
        "harness_reserve_bytes": 0,
        "safety_percent": 85,
        "usable_budget_bytes": usable,
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def _usable_capacity(
    usable_gib: int,
    *,
    gfx_target: str,
) -> GPUMemoryQuotaEvidence:
    """Build consistent evidence whose derived usable quota is exact."""
    raw = usable_gib * GIB * 5 // 4
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": f"mock {gfx_target}",
        "gfx_target": gfx_target,
        "torch_version": "test",
        "hip_version": "test",
        "collected_at": datetime(2026, 8, 15, tzinfo=UTC),
        "runtime_free_bytes": raw,
        "runtime_total_bytes": raw,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": raw,
        "harness_reserve_bytes": 0,
        "safety_percent": 80,
        "usable_budget_bytes": usable_gib * GIB,
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def _target(name: str = "gfx1200", **updates: Any) -> StaticTargetDescriptor:
    target = load_target_descriptor(TARGETS / "isa" / f"{name}.yaml")
    return target.model_copy(update=updates)


def _view(path: Path) -> CorpusTargetViewManifest:
    raw = yaml.safe_load(
        (path / "target-view-manifest.yaml").read_text(encoding="utf-8")
    )
    return CorpusTargetViewManifest.model_validate(raw)


def _entry_artifacts(corpus: CorpusManifest, index: int = 0):
    entry = corpus.entries[index]
    definition = load_json_file(Definition, RELEASE / entry.definition_path)
    raw = yaml.safe_load(
        (RELEASE / entry.generation_rule_path).read_text(encoding="utf-8")
    )
    rule = WorkloadGenerationRule.model_validate(raw)
    facts = next(
        source.architecture_facts
        for source in corpus.sources
        if source.source_id == entry.source_ids[0]
    )
    return entry, definition, rule, facts


def _distribution_signature(view: CorpusTargetViewManifest) -> tuple[Any, ...]:
    """Remove common scale while retaining every frozen distribution point."""
    decisions = tuple(
        (item.semantic_id, item.status, item.distribution_id)
        for item in view.decisions
    )
    slots = tuple(
        (
            item.semantic_id,
            item.slot_id,
            item.role,
            item.regime,
            item.serving_phase,
            item.binding,
            item.scale_numerator,
            item.scale_denominator,
            item.input_profile_id,
            item.input_profile_digest,
            item.correctness_profile_id,
            item.correctness_profile_digest,
        )
        for item in view.workloads
    )
    return decisions, slots


def test_release_freezes_rules_without_concrete_workloads(
    corpus: CorpusManifest,
) -> None:
    report = validate_corpus(MANIFEST)

    assert report["definitions"] == 36
    assert report["generation_rules"] == 36
    assert report["concrete_workloads"] == 0
    assert len(tuple(RELEASE.rglob("generation-rule.yaml"))) == 36
    assert not tuple(RELEASE.rglob("workload.jsonl"))
    assert all(
        len(_entry_artifacts(corpus, i)[2].slots) == 9 for i in range(36)
    )


def test_capacity_class_boundaries() -> None:
    classes = (1, 2, 4, 384)

    assert workload_generation.capacity_class_bytes(GIB - 1, classes) == 0
    assert workload_generation.capacity_class_bytes(GIB, classes) == GIB
    assert workload_generation.capacity_class_bytes(3 * GIB, classes) == 2 * GIB
    assert (
        workload_generation.capacity_class_bytes(500 * GIB, classes)
        == 384 * GIB
    )


def test_same_cohort_ignores_all_target_view_audit_fields(
    tmp_path: Path,
) -> None:
    target = _target()
    renamed_target = target.model_copy(
        update={
            "hardware": target.hardware.model_copy(
                update={"target_id": "audit-only-id"}
            )
        }
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_corpus(
        MANIFEST,
        first,
        target=renamed_target,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(10),
    )
    generate_corpus(
        MANIFEST,
        second,
        target=target,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(
            10,
            collected_at=datetime(2027, 1, 1, tzinfo=UTC),
            gpu_name="different audit name",
            device="cuda:7",
            device_index=7,
        ),
    )

    left, right = _view(first), _view(second)
    assert left.generation_cohort_id == right.generation_cohort_id
    assert left.workload_view_digest == right.workload_view_digest
    assert left.decisions == right.decisions
    assert left.target_descriptor_sha256 != right.target_descriptor_sha256
    assert left.capacity_evidence.capacity_probe_digest != (
        right.capacity_evidence.capacity_probe_digest
    )
    assert [(item.axes, item.uuid) for item in left.workloads] == [
        (item.axes, item.uuid) for item in right.workloads
    ]


def test_same_capacity_class_ignores_exact_free_bytes(tmp_path: Path) -> None:
    target = _target()
    first = tmp_path / "ten"
    second = tmp_path / "eleven"
    for output, free in ((first, 10), (second, 11)):
        generate_corpus(
            MANIFEST,
            output,
            target=target,
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=_capacity(free),
        )

    left, right = _view(first), _view(second)
    assert left.capacity_class_id == right.capacity_class_id == "mem-8gib"
    assert left.capacity_evidence.capacity_probe_digest != (
        right.capacity_evidence.capacity_probe_digest
    )
    assert left.generation_cohort_id == right.generation_cohort_id
    assert left.workload_view_digest == right.workload_view_digest
    assert [(item.axes, item.uuid) for item in left.workloads] == [
        (item.axes, item.uuid) for item in right.workloads
    ]


def test_capability_order_does_not_change_cohort(tmp_path: Path) -> None:
    target = _target()
    reordered = target.model_copy(
        update={"capabilities": tuple(reversed(target.capabilities))}
    )
    outputs = (tmp_path / "normal", tmp_path / "reordered")
    for output, descriptor in zip(outputs, (target, reordered), strict=True):
        generate_corpus(
            MANIFEST,
            output,
            target=descriptor,
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=_capacity(10),
        )

    left, right = map(_view, outputs)
    assert left.generation_cohort_id == right.generation_cohort_id
    assert left.workload_view_digest == right.workload_view_digest


def test_normalized_gfx_spelling_does_not_change_generation(
    tmp_path: Path,
) -> None:
    outputs = (tmp_path / "lower", tmp_path / "upper")
    lower = _target()
    upper_payload = lower.model_dump(mode="python")
    upper_payload["hardware"]["gfx_target"] = " GFX1200 "
    upper = StaticTargetDescriptor.model_validate(upper_payload)
    capacities = (
        _capacity(10),
        _capacity(10, gfx_target=" GFX1200 "),
    )
    for output, target, capacity in zip(
        outputs, (lower, upper), capacities, strict=True
    ):
        generate_corpus(
            MANIFEST,
            output,
            target=target,
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=capacity,
        )

    left, right = map(_view, outputs)
    assert left.generation_cohort_id == right.generation_cohort_id
    assert left.workload_view_digest == right.workload_view_digest
    assert left.decisions == right.decisions
    assert [(item.axes, item.uuid) for item in left.workloads] == [
        (item.axes, item.uuid) for item in right.workloads
    ]


def test_static_target_descriptor_rejects_resolved_device_kind() -> None:
    payload = _target().model_dump(mode="python")
    payload["hardware"]["kind"] = HardwareConfigurationKind.PHYSICAL_DEVICE

    with pytest.raises(ValueError, match="declared template kind"):
        StaticTargetDescriptor.model_validate(payload)


def test_target_view_binds_descriptor_digest_and_resolved_context(
    tmp_path: Path,
) -> None:
    output = tmp_path / "view"
    generate_corpus(
        MANIFEST,
        output,
        target=_target(),
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(10),
    )
    view = _view(output)

    bad_digest = view.model_dump(mode="python")
    bad_digest["target_descriptor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor digest"):
        CorpusTargetViewManifest.model_validate(bad_digest)

    mismatched_target = load_target_descriptor(
        TARGETS / "products/rx9060xt.yaml"
    )
    mismatched_context = view.model_dump(mode="python")
    mismatched_context["target"] = mismatched_target
    mismatched_context["target_descriptor_sha256"] = stable_json_checksum(
        mismatched_target.model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="device model"):
        CorpusTargetViewManifest.model_validate(mismatched_context)


def test_executor_major_version_changes_cohort() -> None:
    target = _target()
    inputs = {
        "distribution_id": "a" * 64,
        "target": target,
        "capacity_class_bytes": 8 * GIB,
        "capacity_numerator": 1,
        "capacity_denominator": 2,
    }

    first = workload_generation.generation_cohort_id(
        **inputs, executor_version="llm_core_common_scale.v1"
    )
    second = workload_generation.generation_cohort_id(
        **inputs, executor_version="llm_core_common_scale.v2"
    )

    assert first != second


def test_distribution_identity_binds_model_facts_and_protocol(
    corpus: CorpusManifest,
    monkeypatch,
) -> None:
    entry, definition, rule, facts = _entry_artifacts(corpus)
    original_input_digest = workload_generation.input_profile_digest
    baseline = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=facts,
    )
    changed_facts = facts.model_copy(
        update={"hidden_size": (facts.hidden_size or 1) + 1}
    )
    model_changed = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=changed_facts,
    )
    protocol_changed = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=facts,
        protocol_major=WORKLOAD_GENERATION_PROTOCOL_MAJOR + 1,
    )
    changed_slot = rule.slots[0].model_copy(update={"scale_numerator": 2})
    rule_changed = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule.model_copy(update={"slots": (changed_slot, *rule.slots[1:])}),
        facts=facts,
    )
    monkeypatch.setattr(
        workload_generation,
        "input_profile_digest",
        lambda _rule, regime: regime.value[0] * 64,
    )
    input_profile_changed = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=facts,
    )
    monkeypatch.setattr(
        workload_generation, "input_profile_digest", original_input_digest
    )
    monkeypatch.setattr(
        workload_generation,
        "correctness_profile_digest",
        lambda _family: "d" * 64,
    )
    correctness_profile_changed = workload_generation.distribution_id(
        semantic_fingerprint=entry.semantic_fingerprint,
        definition=definition,
        rule=rule,
        facts=facts,
    )

    assert (
        len(
            {
                baseline,
                model_changed,
                protocol_changed,
                rule_changed,
                input_profile_changed,
                correctness_profile_changed,
            }
        )
        == 6
    )


def test_workload_uuid_uses_only_frozen_semantic_inputs(
    corpus: CorpusManifest,
) -> None:
    entry, definition, rule, facts = _entry_artifacts(corpus)
    cohort = "c" * 64
    result = workload_generation.generate_rule_workloads(
        definition=definition,
        rule=rule,
        facts=facts,
        target=_target(),
        capacity_bytes=8 * GIB,
        cohort_id=cohort,
        semantic_fingerprint=entry.semantic_fingerprint,
    )

    assert result is not None
    for slot, workload in zip(rule.slots, result.workloads, strict=True):
        canonical_axes = json.dumps(
            workload.axes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        identity = (
            result.distribution_id
            + cohort
            + rule.semantic_id
            + slot.slot_id
            + canonical_axes
        )
        assert workload.uuid == str(
            uuid.uuid5(workload_generation.WORKLOAD_NAMESPACE, identity)
        )


def test_alignment_has_explicit_minimum_distinct_scale(
    corpus: CorpusManifest,
) -> None:
    _, _, rule, facts = _entry_artifacts(corpus)
    minimum = workload_generation._minimum_distinct_scale(rule, facts)

    assert minimum == 48
    axes = [
        workload_generation._slot_axes(rule, facts, slot, minimum - 1)
        for slot in rule.slots
    ]
    assert len({tuple(sorted(row.items())) for row in axes}) < 9


def test_minimum_distinct_above_feasible_skips_entire_definition(
    corpus: CorpusManifest,
) -> None:
    entry, definition, rule, facts = _entry_artifacts(corpus)
    target = _target(max_tensor_bytes=1024, reference_ipc_limit_bytes=4096)

    result = workload_generation.generate_rule_workloads(
        definition=definition,
        rule=rule,
        facts=facts,
        target=target,
        capacity_bytes=8 * GIB,
        cohort_id="a" * 64,
        semantic_fingerprint=entry.semantic_fingerprint,
    )

    assert result is None


def test_context_limit_rejects_whole_scale_without_slot_clamp(
    corpus: CorpusManifest,
) -> None:
    index = next(
        index
        for index, item in enumerate(corpus.entries)
        if item.operation_family is CorpusOperationFamily.ATTENTION
    )
    entry, definition, rule, facts = _entry_artifacts(corpus, index)
    limited_facts = facts.model_copy(update={"maximum_context": 256})
    result = workload_generation.generate_rule_workloads(
        definition=definition,
        rule=rule,
        facts=limited_facts,
        target=_target(),
        capacity_bytes=384 * GIB,
        cohort_id="b" * 64,
        semantic_fingerprint=entry.semantic_fingerprint,
    )

    assert result is not None
    assert [item.axes for item in result.records] == [
        workload_generation._slot_axes(
            rule, limited_facts, slot, result.common_scale
        )
        for slot in rule.slots
    ]
    assert all(
        axes.get("S", 1) <= 256 and axes.get("T", 1) <= 256
        for axes in (item.axes for item in result.records)
    )
    assert not workload_generation._resource_scale_is_feasible(
        definition,
        rule,
        limited_facts,
        _target(),
        192 * GIB,
        result.common_scale + 1,
    )


def test_resource_envelopes_are_recomputed_and_capped_per_workload(
    tmp_path: Path,
    corpus: CorpusManifest,
) -> None:
    output = tmp_path / "view"
    generate_corpus(
        MANIFEST,
        output,
        target=_target(),
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_usable_capacity(8, gfx_target="gfx1200"),
    )
    view = _view(output)
    entries = {item.semantic_id: item for item in corpus.entries}
    grouped_peaks: dict[str, list[int]] = {}
    for record in view.workloads:
        entry = entries[record.semantic_id]
        definition = load_json_file(Definition, RELEASE / entry.definition_path)
        rule = WorkloadGenerationRule.model_validate(
            yaml.safe_load(
                (RELEASE / entry.generation_rule_path).read_text(
                    encoding="utf-8"
                )
            )
        )
        recomputed = workload_generation.workload_requirements(
            definition,
            record.axes,
            rule.operation_family,
            rule.quantization,
            rule.capabilities,
        )
        assert recomputed == record.requirements
        assert record.requirements.resources.reference_peak_bytes <= 4 * GIB
        grouped_peaks.setdefault(record.semantic_id, []).append(
            record.requirements.resources.reference_peak_bytes
        )

    assert any(sum(peaks) > 4 * GIB for peaks in grouped_peaks.values())


def test_mock_hardware_matrix_preserves_distribution(tmp_path: Path) -> None:
    scenarios = (
        (
            "gfx1200-8",
            _target("gfx1200"),
            _usable_capacity(8, gfx_target="gfx1200"),
        ),
        (
            "gfx1200-16",
            _target("gfx1200"),
            _usable_capacity(16, gfx_target="gfx1200"),
        ),
        (
            "gfx942-192",
            _target("gfx942"),
            _usable_capacity(192, gfx_target="gfx942"),
        ),
    )
    views: list[CorpusTargetViewManifest] = []
    for name, target, capacity in scenarios:
        output = tmp_path / name
        generate_corpus(
            MANIFEST,
            output,
            target=target,
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=capacity,
        )
        views.append(_view(output))

    assert [view.capacity_class_bytes for view in views] == [
        8 * GIB,
        16 * GIB,
        192 * GIB,
    ]
    assert len({_distribution_signature(view) for view in views}) == 1
    assert views[0].generation_cohort_id != views[1].generation_cohort_id
    assert views[1].generation_cohort_id != views[2].generation_cohort_id


def test_generated_definition_contains_exact_stable_slots(
    tmp_path: Path,
) -> None:
    generate_corpus(
        MANIFEST,
        tmp_path / "view",
        target=_target(),
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(10),
    )
    view = _view(tmp_path / "view")
    generated = [
        decision
        for decision in view.decisions
        if decision.status is GenerationDecisionStatus.GENERATED
    ]

    assert generated
    assert all(len(decision.workload_uuids) == 9 for decision in generated)
    for decision in generated:
        records = [
            item
            for item in view.workloads
            if item.semantic_id == decision.semantic_id
        ]
        assert len(records) == 9
        assert len({item.slot_id for item in records}) == 9
        assert len({tuple(sorted(item.axes.items())) for item in records}) == 9


def test_minimum_capacity_smoke_references_execute_on_cpu(
    tmp_path: Path,
) -> None:
    output = tmp_path / "minimum"
    generate_corpus(
        MANIFEST,
        output,
        target=_target(),
        profiles=tuple(CorpusProfile),
        capacity_evidence=_usable_capacity(1, gfx_target="gfx1200"),
    )
    view = _view(output)
    smoke_ids = {
        item.uuid for item in view.workloads if item.slot_id == "smoke"
    }

    assert smoke_ids
    for problem in view.problems:
        definition = load_json_file(
            Definition, output / problem.definition_path
        )
        workloads = load_jsonl_file(Workload, output / problem.workload_path)
        smoke = next(item for item in workloads if item.uuid in smoke_ids)
        _, reference = load_reference_function(definition.reference)
        inputs = gen_inputs(definition, smoke, "cpu", seed=0)
        with torch.no_grad():
            result = reference(*inputs)
        assert result is not None


def test_below_minimum_capacity_never_partially_generates(
    tmp_path: Path,
) -> None:
    generate_corpus(
        MANIFEST,
        tmp_path / "view",
        target=_target(),
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(1),
    )
    view = _view(tmp_path / "view")

    assert view.capacity_class_bytes == 0
    assert view.capacity_class_id is None
    assert not view.workloads
    eligible = [
        item
        for item in view.decisions
        if item.status is not GenerationDecisionStatus.PROFILE_NOT_SELECTED
    ]
    assert eligible
    assert all(
        item.status is GenerationDecisionStatus.INSUFFICIENT_CAPACITY
        for item in eligible
    )


def test_generation_failure_does_not_regenerate_or_leave_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = 0

    def fail_generation(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic runtime failure")

    monkeypatch.setattr(
        "sol_execbench.core.dataset.corpus.generate_rule_workloads",
        fail_generation,
    )
    output = tmp_path / "view"
    with pytest.raises(RuntimeError, match="synthetic runtime failure"):
        generate_corpus(
            MANIFEST,
            output,
            target=_target(),
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=_capacity(10),
        )

    assert calls == 1
    assert not output.exists()


def test_existing_view_cannot_be_regenerated_after_runtime_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "view"
    target = _target()
    capacity = _usable_capacity(8, gfx_target="gfx1200")
    generate_corpus(
        MANIFEST,
        output,
        target=target,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=capacity,
    )
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RuntimeError, match="synthetic execution OOM"):
        raise RuntimeError("synthetic execution OOM")
    calls = 0

    def forbidden_regeneration(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("runtime failure must not re-enter generation")

    monkeypatch.setattr(
        "sol_execbench.core.dataset.corpus.generate_rule_workloads",
        forbidden_regeneration,
    )
    with pytest.raises(FileExistsError):
        generate_corpus(
            MANIFEST,
            output,
            target=target,
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=capacity,
        )

    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert calls == 0
    assert after == before
