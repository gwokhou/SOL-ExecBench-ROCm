from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.bench.input_generation import gen_inputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.corpus import (
    load_corpus_manifest,
    load_target_descriptor,
    select_corpus,
    static_selection_reason,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusManifest,
    CorpusProfile,
    CorpusSelectionManifest,
    QuantizationScheme,
    SelectionReason,
    StaticCapability,
    StaticTargetDescriptor,
    TargetQualificationStatus,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V1/manifest.yaml"
TARGETS = ROOT / "problems/LLM_CORE/targets"
GENERATOR = ROOT / "scripts/build_llm_core_corpus.py"


@pytest.fixture(scope="module")
def corpus() -> CorpusManifest:
    return load_corpus_manifest(MANIFEST)


def test_frozen_corpus_meets_release_floors(corpus: CorpusManifest) -> None:
    report = validate_corpus(MANIFEST)

    assert report["status"] == "valid"
    assert report["release_id"] == "LLM_CORE_V1"
    assert report["release_state"] == "frozen"
    assert report["definitions"] == 84
    assert report["workloads"] == 1260
    assert report["sources"] == 8
    assert report["operation_families"] == {
        "advanced_attention": 10,
        "attention": 12,
        "indexing_reduction": 6,
        "kv_cache": 8,
        "linear": 12,
        "moe": 12,
        "norm_activation": 10,
        "position": 6,
        "quantization": 8,
    }


def test_corpus_requires_exact_current_schema() -> None:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    raw.pop("schema_version")

    with pytest.raises(ValueError, match="requires schema_version"):
        CorpusManifest.model_validate(raw)


def test_manifest_rejects_duplicate_semantic_fingerprint(
    corpus: CorpusManifest,
) -> None:
    raw = json.loads(corpus.model_dump_json())
    raw["entries"][1]["semantic_fingerprint"] = raw["entries"][0][
        "semantic_fingerprint"
    ]

    with pytest.raises(ValidationError, match="fingerprints must be unique"):
        CorpusManifest.model_validate(raw)


def test_all_sources_are_pinned_and_clean_room(corpus: CorpusManifest) -> None:
    assert len({source.source_id for source in corpus.sources}) == 8
    assert all(len(source.revision) == 40 for source in corpus.sources)
    assert all(source.license_reviewed for source in corpus.sources)
    assert all(source.clean_room for source in corpus.sources)
    assert "nvidia/SOL-ExecBench" not in MANIFEST.read_text(encoding="utf-8")


def test_every_definition_has_cpu_micro_reference(
    corpus: CorpusManifest,
) -> None:
    root = MANIFEST.parent
    for entry in corpus.entries:
        definition = load_json_file(Definition, root / entry.definition_path)
        workloads = load_jsonl_file(Workload, root / entry.workload_path)
        inputs = gen_inputs(definition, workloads[0], "cpu", seed=0)
        _module, reference = load_reference_function(definition.reference)
        with torch.inference_mode():
            output = reference(*inputs)
        expected_shape = definition.get_output_shapes(workloads[0].axes)[
            "output"
        ]
        assert isinstance(output, torch.Tensor), entry.semantic_id
        assert output.shape == expected_shape, entry.semantic_id


def _target(**updates: object) -> StaticTargetDescriptor:
    payload = {
        "schema_version": SchemaVersion.STATIC_TARGET_DESCRIPTOR,
        "target_id": "synthetic",
        "gfx_target": "gfx-test",
        "qualification_status": TargetQualificationStatus.DECLARED,
        "declaration_source": "unit test",
        "memory_budget_bytes": 1 << 50,
        "max_tensor_bytes": 1 << 50,
        "reference_ipc_limit_bytes": 1 << 50,
        "supported_dtypes": list(DType),
        "supported_quantization": list(QuantizationScheme),
        "capabilities": list(StaticCapability),
    }
    payload.update(updates)
    return StaticTargetDescriptor.model_validate(payload)


def test_static_filter_reason_order(corpus: CorpusManifest) -> None:
    core_entry = corpus.entries[0]
    core_workload = core_entry.workloads[0]
    quant_entry = next(
        entry
        for entry in corpus.entries
        if CorpusProfile.QUANTIZED in entry.profiles
    )
    quant_workload = quant_entry.workloads[0]

    cases = (
        (
            core_entry,
            core_workload,
            _target(),
            frozenset((CorpusProfile.MOE,)),
            SelectionReason.PROFILE_NOT_SELECTED,
        ),
        (
            core_entry,
            core_workload,
            _target(supported_dtypes=[DType.FLOAT32]),
            frozenset((CorpusProfile.CORE,)),
            SelectionReason.UNSUPPORTED_DTYPE,
        ),
        (
            quant_entry,
            quant_workload,
            _target(supported_quantization=[]),
            frozenset((CorpusProfile.QUANTIZED,)),
            SelectionReason.UNSUPPORTED_QUANTIZATION,
        ),
        (
            core_entry,
            core_workload,
            _target(capabilities=[]),
            frozenset((CorpusProfile.CORE,)),
            SelectionReason.MISSING_CAPABILITY,
        ),
        (
            core_entry,
            core_workload,
            _target(max_tensor_bytes=1),
            frozenset((CorpusProfile.CORE,)),
            SelectionReason.TENSOR_LIMIT_EXCEEDED,
        ),
        (
            core_entry,
            core_workload,
            _target(reference_ipc_limit_bytes=1),
            frozenset((CorpusProfile.CORE,)),
            SelectionReason.REFERENCE_IPC_LIMIT_EXCEEDED,
        ),
        (
            core_entry,
            core_workload,
            _target(memory_budget_bytes=1),
            frozenset((CorpusProfile.CORE,)),
            SelectionReason.MEMORY_BUDGET_EXCEEDED,
        ),
    )
    for entry, workload, target, profiles, expected in cases:
        reason, _detail = static_selection_reason(
            entry,
            workload,
            target,
            profiles,
        )
        assert reason is expected


def _included(
    corpus: CorpusManifest,
    target: StaticTargetDescriptor,
) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    profiles = frozenset(CorpusProfile)
    for entry in corpus.entries:
        for workload in entry.workloads:
            reason, _detail = static_selection_reason(
                entry,
                workload,
                target,
                profiles,
            )
            if reason is SelectionReason.INCLUDED:
                selected.add((entry.semantic_id, workload.uuid))
    return selected


def test_larger_memory_budget_is_a_selection_superset(
    corpus: CorpusManifest,
) -> None:
    small = _included(corpus, _target(memory_budget_bytes=2 * 1024**3))
    large = _included(corpus, _target(memory_budget_bytes=32 * 1024**3))

    assert small
    assert small <= large
    assert small != large


def test_static_selection_materializes_auditable_view(
    corpus: CorpusManifest,
    tmp_path: Path,
) -> None:
    output = tmp_path / "selected"
    result = select_corpus(
        MANIFEST,
        output,
        target=_target(memory_budget_bytes=8 * 1024**3),
        profiles=(CorpusProfile.CORE, CorpusProfile.QUANTIZED),
    )

    record = CorpusSelectionManifest.model_validate(
        yaml.safe_load((result / "selection-manifest.yaml").read_text()),
    )
    assert (
        record.target.qualification_status is TargetQualificationStatus.DECLARED
    )
    assert record.coverage["workloads"] > 0
    assert record.problems
    assert all(decision.detail for decision in record.decisions)
    for problem in record.problems:
        assert (result / problem.definition_path).is_file()
        selected = load_jsonl_file(Workload, result / problem.workload_path)
        assert tuple(item.uuid for item in selected) == problem.workload_uuids


@pytest.mark.parametrize("target_name", ["gfx1200", "gfx942"])
def test_bundled_targets_are_declared_only(target_name: str) -> None:
    target = load_target_descriptor(TARGETS / f"{target_name}.yaml")

    assert target.gfx_target == target_name
    assert target.qualification_status is TargetQualificationStatus.DECLARED
    assert target.memory_budget_bytes is None


def test_selection_rejects_premature_hardware_qualification(
    tmp_path: Path,
) -> None:
    target = _target(
        qualification_status=TargetQualificationStatus.HARDWARE_QUALIFIED,
    )

    with pytest.raises(
        ValueError, match="hardware-qualified targets are deferred"
    ):
        select_corpus(
            MANIFEST,
            tmp_path / "selected",
            target=target,
            profiles=(CorpusProfile.CORE,),
        )


def test_generator_is_byte_deterministic() -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        check=True,
        timeout=30,
    )
