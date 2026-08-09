"""Real handlers derive their stage_id from the identity family.

``test_orchestrator`` drives the chain with fake handlers, so the real
``CorpusSnapshotHandler`` and ``AcceptanceHandler`` stage_id derivation is
not exercised there. These tests pin the runtime contract that every real
handler routes its stage_id through the identity functions rather than
through an ad-hoc digest of its outputs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    acceptance_id,
    corpus_snapshot_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.orchestrator import (
    AcceptanceHandler,
    CollectionRunHandler,
    CorpusSnapshotHandler,
    StageRunContext,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    stage_receipt_path,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file


def test_collection_handler_requires_exact_reviewed_inventory(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "collection"
    corpus_root.mkdir()
    corpus = corpus_root / "held_out.json"
    corpus.write_text("held out", encoding="utf-8")
    artifact = DiagnosticLifecycleArtifact(
        relative_path=corpus.name,
        sha256=sha256_file(corpus),
        size_bytes=corpus.stat().st_size,
    )
    context = StageRunContext(
        store_root=tmp_path,
        plan=cast(
            Any,
            SimpleNamespace(
                collection_inventory=(artifact,),
                held_out_corpus=artifact,
            ),
        ),
        design_manifest_path=tmp_path / "design.json",
        collection_run_id="a" * 64,
        generation=1,
        corpus_root=corpus_root,
        held_out_corpus_path=corpus,
    )
    handler = CollectionRunHandler()
    completion = handler.run(context)
    receipt = DiagnosticStageReceipt(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        stage_id=completion.stage_id,
        command="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        attempts=1,
        output_inventory=completion.outputs,
    )
    assert handler.verify(context, receipt)

    (corpus_root / "extra.json").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reviewed plan"):
        handler.run(context)
    assert not handler.verify(context, receipt)


def test_corpus_snapshot_handler_derives_identity(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    held_out = corpus_root / "held_out.json"
    held_out.write_text("held out corpus", encoding="utf-8")
    collection_run_id = "a" * 64
    held_out_artifact = DiagnosticLifecycleArtifact(
        relative_path="held_out.json",
        sha256=sha256_file(held_out),
        size_bytes=held_out.stat().st_size,
    )
    context = StageRunContext(
        store_root=tmp_path,
        plan=cast(Any, SimpleNamespace(held_out_corpus=held_out_artifact)),
        design_manifest_path=tmp_path / "design.json",
        collection_run_id=collection_run_id,
        generation=1,
        corpus_root=corpus_root,
        held_out_corpus_path=held_out,
    )

    completion = CorpusSnapshotHandler().run(context)

    assert completion.stage_id == corpus_snapshot_id(
        collection_run_id=collection_run_id,
        role="held_out",
        corpus_sha256=sha256_file(held_out),
        source_revision="unknown",
    )


def test_acceptance_handler_derives_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_run_id = "a" * 64
    model_build_id = "b" * 64
    calibration_id = "c" * 64
    development_snapshot_id = "d" * 64
    store = tmp_path
    receipt = DiagnosticStageReceipt(
        stage=DiagnosticLifecycleStage.MODEL_BUILD,
        stage_id=model_build_id,
        command="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        attempts=1,
    )
    atomic_write_json_value(
        stage_receipt_path(
            collection_run_id,
            DiagnosticLifecycleStage.MODEL_BUILD,
            store,
        ),
        receipt.model_dump(mode="json"),
    )
    for stage, stage_id in (
        (DiagnosticLifecycleStage.CALIBRATION, calibration_id),
        (DiagnosticLifecycleStage.CORPUS_SNAPSHOT, development_snapshot_id),
    ):
        dependency_receipt = receipt.model_copy(
            update={"stage": stage, "stage_id": stage_id}
        )
        atomic_write_json_value(
            stage_receipt_path(collection_run_id, stage, store),
            dependency_receipt.model_dump(mode="json"),
        )
    held_out = tmp_path / "held_out.json"
    held_out.write_text("held out", encoding="utf-8")
    held_out_sha = sha256_file(held_out)
    output_root = tmp_path / "out"
    fake_manifest = SimpleNamespace(
        held_out_corpus_sha256=held_out_sha,
        model_dump=lambda **kwargs: {"manifest": 1},
    )
    fake_result = SimpleNamespace(
        accepted=True,
        model_dump=lambda **kwargs: {"result": 1},
    )
    monkeypatch.setattr(
        "sol_execbench.core.bench.performance_model.authoring.build_diagnostic_acceptance",
        lambda **kwargs: (fake_manifest, fake_result),
    )
    context = StageRunContext(
        store_root=store,
        plan=cast(
            Any,
            SimpleNamespace(
                development_snapshot=DiagnosticLifecycleParent(
                    stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
                    stage_id=development_snapshot_id,
                    sha256="e" * 64,
                )
            ),
        ),
        design_manifest_path=store / "design.json",
        collection_run_id=collection_run_id,
        generation=1,
        calibration_profile_path=tmp_path / "calibration.json",
        development_corpus_path=tmp_path / "development.json",
        held_out_corpus_path=held_out,
        output_root=output_root,
    )
    context.set_output(
        DiagnosticLifecycleStage.MODEL_BUILD,
        tmp_path / "inference.json",
    )

    handler = AcceptanceHandler(semantic_loader=cast(Any, object()))
    completion = handler.run(context)

    expected = acceptance_id(
        calibration_id=calibration_id,
        development_snapshot_id=development_snapshot_id,
        model_build_id=model_build_id,
        held_out_corpus_snapshot_id=corpus_snapshot_id(
            collection_run_id=collection_run_id,
            role="held_out",
            corpus_sha256=held_out_sha,
            source_revision="unknown",
        ),
        accepted=True,
        verdict_sha256=sha256_file(
            output_root / "acceptance" / "acceptance.json"
        ),
        source_revision="unknown",
    )
    assert completion.stage_id == expected
