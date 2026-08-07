from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticLifecycleStage,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    publication_registry_dir,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

ScriptLoader = Callable[[str], ModuleType]


def _publication_manifest(
    stage_id: str,
    digest: str,
    size: int,
) -> DiagnosticPublicationLifecycleManifest:
    return DiagnosticPublicationLifecycleManifest(
        stage=DiagnosticLifecycleStage.PUBLICATION,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision="test",
        created_at="2026-01-01T00:00:00+00:00",
        publication_manifest_sha256=digest,
        uncompressed_size_bytes=size,
        case_count=220,
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                sha256=digest,
                size_bytes=size,
            ),
        ),
    )


def test_consistent_store_reports_no_findings(
    load_script: ScriptLoader,
    tmp_path: Path,
) -> None:
    script = load_script("scripts/check_diagnostic_store_consistency.py")
    store = BlobStore(tmp_path)
    digest = store.put_bytes(b"evidence")
    manifest = _publication_manifest("p" * 64, digest, 9)
    path = (
        publication_registry_dir(tmp_path) / manifest.stage_id / "manifest.json"
    )
    atomic_write_json_value(path, manifest.model_dump(mode="json"))

    assert script.check_store(tmp_path) == []


def test_inconsistent_store_flags_placement_and_missing_blob(
    load_script: ScriptLoader,
    tmp_path: Path,
) -> None:
    script = load_script("scripts/check_diagnostic_store_consistency.py")
    BlobStore(tmp_path).put_bytes(b"evidence")
    manifest = _publication_manifest("p" * 64, "f" * 64, 9)
    wrong_dir = tmp_path / "designs" / "WRONG" / "manifest.json"
    atomic_write_json_value(wrong_dir, manifest.model_dump(mode="json"))

    findings = script.check_store(tmp_path)

    assert any("does not match directory" in item for item in findings)
    assert any("does not match stage_id" in item for item in findings)
    assert any("missing blob" in item for item in findings)
