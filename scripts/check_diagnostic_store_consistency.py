#!/usr/bin/env python3
"""Fail when the diagnostic lifecycle store is internally inconsistent.

The lifecycle store is the authoritative registry for the diagnostic chain.
Every lifecycle manifest must live under ``<store>/<stage_dir>/<stage_id>/
manifest.json`` with a directory and ``stage_id`` that match its own
``stage`` and ``stage_id``, its stored ``stage_id`` must recompute from its
own identity inputs (parents and manifest fields), and every digest it
cites (inventory, parents, and stage-specific SHA fields) must exist in the
content-addressed blob store *and* verify its content against its key.
Run-state objects and typed receipts must likewise cite blobs that exist and
verify. A missing store is not an error; a store that contradicts its own
registry layout, identity, or blobs is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    recompute_stage_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    acceptances_dir,
    builds_dir,
    calibrations_dir,
    designs_dir,
    orchestrations_dir,
    publication_registry_dir,
    releases_dir,
    runs_dir,
    snapshots_dir,
    store_root,
)
from sol_execbench.core.integrity import sha256_file

_DIR_TO_STAGE: Final[dict[str, DiagnosticLifecycleStage]] = {
    "designs": DiagnosticLifecycleStage.DESIGN,
    "collection-runs": DiagnosticLifecycleStage.COLLECTION_RUN,
    "snapshots": DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
    "calibrations": DiagnosticLifecycleStage.CALIBRATION,
    "builds": DiagnosticLifecycleStage.MODEL_BUILD,
    "acceptances": DiagnosticLifecycleStage.ACCEPTANCE,
    "publication-registry": DiagnosticLifecycleStage.PUBLICATION,
    "release-candidates": DiagnosticLifecycleStage.RELEASE,
}

_SHA_FIELDS: Final[tuple[str, ...]] = (
    "design_payload_sha256",
    "frozen_held_out_sha256",
    "corpus_file_sha256",
    "calibration_profile_sha256",
    "calibration_audit_sha256",
    "inference_profile_sha256",
    "verdict_sha256",
    "source_corpus_sha256",
    "publication_manifest_sha256",
    "archive_sha256",
    "attestation_sha256",
)


def _blob_exists(root: Path, digest: str) -> bool:
    """Return whether the blob exists and its content verifies to its key."""
    return BlobStore(root).contains(digest)


def _check_manifest(root: Path, manifest_path: Path) -> list[str]:
    directory = manifest_path.parents[1].name
    stage_id_dir = manifest_path.parent.name
    if directory not in _DIR_TO_STAGE:
        return [f"unexpected registry directory: {manifest_path}"]
    try:
        manifest = DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_json(
            manifest_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError) as error:
        return [f"unreadable lifecycle manifest {manifest_path}: {error}"]
    expected_stage = _DIR_TO_STAGE[directory]
    findings: list[str] = []
    if manifest.stage is not expected_stage:
        findings.append(
            f"{manifest_path}: stage {manifest.stage.value!r} does not match "
            f"directory {directory!r}",
        )
    if stage_id_dir != manifest.stage_id:
        findings.append(
            f"{manifest_path}: directory {stage_id_dir!r} does not match "
            f"stage_id {manifest.stage_id!r}",
        )
    try:
        expected_stage_id = recompute_stage_id(manifest)
    except ValueError as error:
        findings.append(
            f"{manifest_path}: cannot recompute stage_id: {error}",
        )
    else:
        if expected_stage_id != manifest.stage_id:
            findings.append(
                f"{manifest_path}: stored stage_id {manifest.stage_id!r} "
                f"does not match recomputed identity "
                f"{expected_stage_id!r}",
            )
    digests: set[str] = set()
    for item in manifest.exact_inventory:
        digests.add(item.sha256)
    for parent in manifest.parents:
        digests.add(parent.sha256)
        if parent.purpose is not manifest.purpose:
            findings.append(
                f"{manifest_path}: parent {parent.stage_id} crosses purpose"
            )
        parent_path = _parent_manifest_path(root, parent.stage, parent.stage_id)
        if not parent_path.is_file():
            findings.append(
                f"{manifest_path}: missing parent manifest {parent_path}"
            )
        elif sha256_file(parent_path) != parent.sha256:
            findings.append(
                f"{manifest_path}: parent manifest digest mismatch "
                f"for {parent.stage_id}"
            )
    for field in _SHA_FIELDS:
        value = getattr(manifest, field, None)
        if isinstance(value, str) and value:
            digests.add(value)
    for digest in sorted(digests):
        if not _blob_exists(root, digest):
            findings.append(
                f"{manifest_path}: missing blob referenced by {digest}",
            )
    return findings


def _parent_manifest_path(
    root: Path,
    stage: DiagnosticLifecycleStage,
    stage_id: str,
) -> Path:
    directories = {
        DiagnosticLifecycleStage.DESIGN: designs_dir(root),
        DiagnosticLifecycleStage.COLLECTION_RUN: runs_dir(root),
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT: snapshots_dir(root),
        DiagnosticLifecycleStage.CALIBRATION: calibrations_dir(root),
        DiagnosticLifecycleStage.MODEL_BUILD: builds_dir(root),
        DiagnosticLifecycleStage.ACCEPTANCE: acceptances_dir(root),
        DiagnosticLifecycleStage.PUBLICATION: publication_registry_dir(root),
        DiagnosticLifecycleStage.RELEASE: releases_dir(root),
    }
    return directories[stage] / stage_id / "manifest.json"


def _check_run_state(root: Path, run_state_path: Path) -> list[str]:
    try:
        run_state = DiagnosticRunManifest.model_validate_json(
            run_state_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError) as error:
        return [f"unreadable run-state {run_state_path}: {error}"]
    findings: list[str] = []
    if run_state_path.parent.name != run_state.collection_run_id:
        findings.append(
            f"{run_state_path}: directory does not match collection_run_id",
        )
    for state in run_state.stages:
        for item in state.outputs:
            if not _blob_exists(root, item.sha256):
                findings.append(
                    f"{run_state_path}: missing blob {item.sha256} "
                    f"referenced by {state.stage.value} outputs",
                )
    return findings


def _check_receipt(root: Path, receipt_path: Path) -> list[str]:
    try:
        receipt = DiagnosticStageReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError) as error:
        return [f"unreadable stage receipt {receipt_path}: {error}"]
    findings: list[str] = []
    for item in receipt.output_inventory:
        if not _blob_exists(root, item.sha256):
            findings.append(
                f"{receipt_path}: missing blob {item.sha256} in outputs",
            )
    for parent in receipt.input_identities:
        if not _blob_exists(root, parent.sha256):
            findings.append(
                f"{receipt_path}: missing blob {parent.sha256} in inputs",
            )
        parent_path = _parent_manifest_path(root, parent.stage, parent.stage_id)
        if not parent_path.is_file():
            findings.append(
                f"{receipt_path}: missing parent manifest {parent_path}"
            )
        elif sha256_file(parent_path) != parent.sha256:
            findings.append(
                f"{receipt_path}: parent manifest digest mismatch "
                f"for {parent.stage_id}"
            )
    return findings


def check_store(root: Path) -> list[str]:
    """Return every consistency finding for one lifecycle store."""
    findings: list[str] = []
    for directory in (
        designs_dir(root),
        runs_dir(root),
        snapshots_dir(root),
        calibrations_dir(root),
        builds_dir(root),
        acceptances_dir(root),
        publication_registry_dir(root),
        releases_dir(root),
    ):
        for manifest_path in sorted(directory.glob("*/manifest.json")):
            findings.extend(_check_manifest(root, manifest_path))
    known = {
        path.name
        for path in (
            designs_dir(root),
            runs_dir(root),
            snapshots_dir(root),
            calibrations_dir(root),
            builds_dir(root),
            acceptances_dir(root),
            publication_registry_dir(root),
            releases_dir(root),
            orchestrations_dir(root),
        )
    }
    ignored = {"blobs", "locks", "attempts", "published-releases", "promotions"}
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in known | ignored:
                findings.append(f"unexpected registry directory: {child}")
    for run_state_path in sorted(orchestrations_dir(root).glob("*/run.json")):
        findings.extend(_check_run_state(root, run_state_path))
    for receipt_path in sorted(
        orchestrations_dir(root).glob("*/receipts/*.json")
    ):
        findings.extend(_check_receipt(root, receipt_path))
    return findings


def main() -> int:
    """Report lifecycle store inconsistencies and return a CI exit code."""
    root = store_root()
    if not root.exists():
        print(f"diagnostic store absent at {root}; nothing to check")
        return 0
    findings = check_store(root)
    if findings:
        print("diagnostic lifecycle store is inconsistent:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"diagnostic lifecycle store is consistent at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
