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
    publication_id,
    publication_registry_dir,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

ScriptLoader = Callable[[str], ModuleType]


def _publication_manifest(
    stage_id: str,
    manifest_digest: str,
    source_digest: str,
    size: int,
) -> DiagnosticPublicationLifecycleManifest:
    return DiagnosticPublicationLifecycleManifest(
        stage=DiagnosticLifecycleStage.PUBLICATION,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision="test",
        created_at="2026-01-01T00:00:00+00:00",
        source_corpus_sha256=source_digest,
        publication_manifest_sha256=manifest_digest,
        uncompressed_size_bytes=size,
        case_count=220,
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                sha256=manifest_digest,
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
    manifest_digest = store.put_bytes(b"manifest")
    source_digest = store.put_bytes(b"source")
    stage_id = publication_id(
        source_corpus_sha256=source_digest,
        publication_manifest_sha256=manifest_digest,
        uncompressed_size_bytes=9,
        case_count=220,
    )
    manifest = _publication_manifest(
        stage_id, manifest_digest, source_digest, 9
    )
    path = (
        publication_registry_dir(tmp_path) / manifest.stage_id / "manifest.json"
    )
    atomic_write_json_value(path, manifest.model_dump(mode="json"))

    assert script.check_store(tmp_path) == []


def test_inconsistent_store_flags_placement_identity_and_missing_blob(
    load_script: ScriptLoader,
    tmp_path: Path,
) -> None:
    script = load_script("scripts/check_diagnostic_store_consistency.py")
    store = BlobStore(tmp_path)
    store.put_bytes(b"manifest")
    source_digest = store.put_bytes(b"source")
    # stage_id is a arbitrary literal that cannot recompute from the manifest
    # inputs; placement is also wrong (designs/ instead of publication-registry/).
    manifest = _publication_manifest("p" * 64, "f" * 64, source_digest, 9)
    wrong_dir = tmp_path / "designs" / "WRONG" / "manifest.json"
    atomic_write_json_value(wrong_dir, manifest.model_dump(mode="json"))

    findings = script.check_store(tmp_path)

    assert any("does not match directory" in item for item in findings)
    assert any("does not match stage_id" in item for item in findings)
    assert any(
        "does not match recomputed identity" in item for item in findings
    )
    assert any("missing blob" in item for item in findings)


def test_mutated_blob_content_is_detected(
    load_script: ScriptLoader,
    tmp_path: Path,
) -> None:
    """A blob whose content no longer matches its key fails the content gate."""
    script = load_script("scripts/check_diagnostic_store_consistency.py")
    store = BlobStore(tmp_path)
    manifest_digest = store.put_bytes(b"manifest")
    source_digest = store.put_bytes(b"source")
    # Corrupt the manifest blob in place so the path still exists but the
    # SHA-256 no longer matches the stored digest.
    blob_path = tmp_path / "blobs" / "sha256" / manifest_digest
    blob_path.write_bytes(b"tampered")
    stage_id = publication_id(
        source_corpus_sha256=source_digest,
        publication_manifest_sha256=manifest_digest,
        uncompressed_size_bytes=9,
        case_count=220,
    )
    manifest = _publication_manifest(
        stage_id, manifest_digest, source_digest, 9
    )
    path = (
        publication_registry_dir(tmp_path) / manifest.stage_id / "manifest.json"
    )
    atomic_write_json_value(path, manifest.model_dump(mode="json"))

    findings = script.check_store(tmp_path)
    # The tampered blob is referenced by the manifest inventory and is now
    # reported as missing because its content no longer verifies to its key.
    assert any("missing blob" in item for item in findings)
