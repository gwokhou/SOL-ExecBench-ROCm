from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)


def _digest(label: str, index: int) -> str:
    return stable_json_checksum([label, index])


def _development_corpus() -> DiagnosticValidationCorpus:
    cases: list[DiagnosticValidationCase] = []
    index = 0
    for phase in ("point_fit", "conformal"):
        for family in _FAMILIES:
            for position in range(20):
                cases.append(
                    DiagnosticValidationCase(
                        case_id=f"{phase}-{family.value}-{position:02d}",
                        pair_id=_digest("pair", index),
                        workload_kind=family,
                        evidence_manifest=BlobArtifactReference(
                            sha256=_digest("evidence", index),
                            size_bytes=1,
                            tree_manifest_sha256="f" * 64,
                        ),
                        solar_manifest=BlobArtifactReference(
                            sha256=_digest("solar", index),
                            size_bytes=1,
                            tree_manifest_sha256="f" * 64,
                        ),
                    )
                )
                index += 1
    return DiagnosticValidationCorpus(role="development", cases=cases)


def _register_direct_snapshot(module, root: Path) -> str:
    corpus = _development_corpus()
    corpus_path = root / "development.json"
    atomic_write_json_value(corpus_path, corpus.model_dump(mode="json"))
    run_id = "1" * 64
    run = DiagnosticCollectionRunManifest(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        stage_id=run_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PROCESS_EVIDENCE,
        source_revision="source",
        created_at="2026-01-01T00:00:00+00:00",
        roles=("development",),
        generation=1,
    )
    run_path = module.runs_dir() / run_id / "manifest.json"
    atomic_write_json_value(run_path, run.model_dump(mode="json"))
    snapshot_id = "2" * 64
    snapshot = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id=snapshot_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision="source",
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.COLLECTION_RUN,
                stage_id=run_id,
                sha256=sha256_file(run_path),
            ),
        ),
        created_at="2026-01-01T00:00:00+00:00",
        role="development",
        corpus_file_sha256=sha256_file(corpus_path),
        case_count=440,
    )
    snapshot_path = module.snapshots_dir() / snapshot_id / "manifest.json"
    atomic_write_json_value(snapshot_path, snapshot.model_dump(mode="json"))
    return snapshot_id


def test_registers_exact_phase_sources_idempotently(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script(
        "scripts/internal/rdna4/register_rdna4_development_sources.py"
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE", str(tmp_path / "store")
    )
    monkeypatch.setattr(
        module, "snapshot_blob_inventory", lambda *args, **kwargs: ()
    )
    snapshot_id = _register_direct_snapshot(module, tmp_path)
    arguments = [
        "--root",
        str(tmp_path),
        "--development-snapshot-id",
        snapshot_id,
    ]

    assert module.main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert module.main(arguments) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    point_fit = DiagnosticValidationCorpus.model_validate_json(
        (tmp_path / "development-point-fit.json").read_text(encoding="utf-8")
    )
    conformal = DiagnosticValidationCorpus.model_validate_json(
        (tmp_path / "development-conformal.json").read_text(encoding="utf-8")
    )
    assert len(point_fit.cases) == len(conformal.cases) == 220
    point_pairs = {case.pair_id for case in point_fit.cases}
    conformal_pairs = {case.pair_id for case in conformal.cases}
    assert not point_pairs & conformal_pairs
    assert (
        first["point_fit"]["snapshot_id"] != first["conformal"]["snapshot_id"]
    )


def test_rejects_non_preregistered_phase_case(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script(
        "scripts/internal/rdna4/register_rdna4_development_sources.py"
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE", str(tmp_path / "store")
    )
    snapshot_id = _register_direct_snapshot(module, tmp_path)
    corpus_path = tmp_path / "development.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    payload["cases"][0]["case_id"] = "unknown-elementwise-00"
    atomic_write_json_value(corpus_path, payload)
    snapshot_path = module.snapshots_dir() / snapshot_id / "manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["corpus_file_sha256"] = sha256_file(corpus_path)
    atomic_write_json_value(snapshot_path, snapshot)

    with pytest.raises(ValueError, match="no unique phase"):
        module.main(
            ["--root", str(tmp_path), "--development-snapshot-id", snapshot_id]
        )
