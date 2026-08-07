# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Registry-driven blob garbage collection.

Blob reachability is computed only from the immutable lifecycle registry:
every manifest, run-state object, and typed receipt under the store. A blob
referenced by any non-superseded lifecycle object is retained; a blob
referenced only by superseded generations or by nothing is reclaimable.
Deletion requires an explicit request and re-verifies reachability
immediately before removing any blob.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
    DiagnosticLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    acceptances_dir,
    blob_path,
    builds_dir,
    designs_dir,
    publication_registry_dir,
    releases_dir,
    runs_dir,
    snapshots_dir,
    store_root,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.integrity import SHA256Digest

_RETENTION_PRIORITY: Final[tuple[DiagnosticRetentionClass, ...]] = (
    DiagnosticRetentionClass.CACHE,
    DiagnosticRetentionClass.DEBUG,
    DiagnosticRetentionClass.PROCESS_EVIDENCE,
    DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
    DiagnosticRetentionClass.PUBLICATION_RELEASE,
)

_SHA_FIELDS: Final[tuple[str, ...]] = (
    "design_payload_sha256",
    "frozen_held_out_sha256",
    "corpus_file_sha256",
    "calibration_profile_sha256",
    "calibration_audit_sha256",
    "inference_profile_sha256",
    "verdict_sha256",
    "publication_manifest_sha256",
    "archive_sha256",
    "attestation_sha256",
)


class GCRefusedError(ValueError):
    """Raised when a requested GC deletion would touch a reachable blob."""


class GCEntry(FrozenArtifactModel):
    """One blob's retention decision in a GC plan."""

    digest: SHA256Digest
    size_bytes: int = Field(ge=0)
    retained: bool
    retention_class: DiagnosticRetentionClass
    reason: str = Field(min_length=1)


class GCPlan(FrozenArtifactModel):
    """One dry-run or executed blob retention plan for a store."""

    store_root: str = Field(min_length=1)
    total_blobs: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    retained_bytes: int = Field(ge=0)
    reclaimable_count: int = Field(ge=0)
    reclaimable_bytes: int = Field(ge=0)
    entries: tuple[GCEntry, ...] = ()


def compute_reachable_blobs(
    store_root_path: Path | None = None,
) -> tuple[set[str], set[str]]:
    """Return (live, superseded-only) blob digests referenced by the registry.

    ``live`` contains digests referenced by any non-superseded lifecycle
    object, run-state object, or typed receipt. ``superseded`` contains
    digests referenced by at least one superseded generation.
    """
    live, superseded, _ = _reachability(
        Path(store_root_path).resolve() if store_root_path else store_root(),
    )
    return live, superseded


def _reachability(
    root: Path,
) -> tuple[set[str], set[str], dict[str, DiagnosticRetentionClass]]:
    live: set[str] = set()
    superseded: set[str] = set()
    referrers: dict[str, DiagnosticRetentionClass] = {}
    for directory in (
        designs_dir(root),
        runs_dir(root),
        snapshots_dir(root),
        builds_dir(root),
        acceptances_dir(root),
        publication_registry_dir(root),
        releases_dir(root),
    ):
        for manifest_path in sorted(directory.glob("*/manifest.json")):
            manifest = _load_manifest(manifest_path)
            if manifest is None:
                continue
            is_superseded = manifest.status is DiagnosticStageStatus.SUPERSEDED
            digests, retention = _manifest_digests(manifest)
            for digest in digests:
                if is_superseded:
                    superseded.add(digest)
                else:
                    live.add(digest)
                    _record_referrer(referrers, digest, retention)
    for run_state_path in sorted(runs_dir(root).glob("*/run.json")):
        run_state = _load_run_state(run_state_path)
        if run_state is None:
            continue
        for state in run_state.stages:
            for item in state.outputs:
                live.add(item.sha256)
                _record_referrer(
                    referrers,
                    item.sha256,
                    DiagnosticRetentionClass.PROCESS_EVIDENCE,
                )
    for receipt_path in sorted(runs_dir(root).glob("*/receipts/*.json")):
        receipt = _load_receipt(receipt_path)
        if receipt is None:
            continue
        for item in receipt.output_inventory:
            live.add(item.sha256)
            _record_referrer(
                referrers,
                item.sha256,
                DiagnosticRetentionClass.PROCESS_EVIDENCE,
            )
        for parent in receipt.input_identities:
            live.add(parent.sha256)
            _record_referrer(
                referrers,
                parent.sha256,
                DiagnosticRetentionClass.PROCESS_EVIDENCE,
            )
    return live, superseded, referrers


def _load_manifest(
    path: Path,
) -> DiagnosticLifecycleManifest | None:
    try:
        return DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None


def _load_run_state(path: Path) -> DiagnosticRunManifest | None:
    try:
        return DiagnosticRunManifest.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None


def _load_receipt(path: Path) -> DiagnosticStageReceipt | None:
    try:
        return DiagnosticStageReceipt.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None


def _manifest_digests(
    manifest: DiagnosticLifecycleManifest,
) -> tuple[set[str], DiagnosticRetentionClass]:
    digests: set[str] = set()
    for item in manifest.exact_inventory:
        digests.add(item.sha256)
    for parent in manifest.parents:
        digests.add(parent.sha256)
    for field in _SHA_FIELDS:
        value = getattr(manifest, field, None)
        if isinstance(value, str) and value:
            digests.add(value)
    return digests, manifest.retention_class


def _record_referrer(
    referrers: dict[str, DiagnosticRetentionClass],
    digest: str,
    retention: DiagnosticRetentionClass,
) -> None:
    current = referrers.get(digest)
    if current is None or _RETENTION_PRIORITY.index(
        retention
    ) > _RETENTION_PRIORITY.index(current):
        referrers[digest] = retention


def plan_gc(store_root_path: Path | None = None) -> GCPlan:
    """Compute the blob retention plan for one store without deleting."""
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    live, superseded, referrers = _reachability(root)
    store = BlobStore(root)
    entries: list[GCEntry] = []
    for digest in store.iter_digests():
        path = blob_path(digest, root)
        size = path.stat().st_size if path.is_file() else 0
        if digest in live:
            entries.append(
                GCEntry(
                    digest=digest,
                    size_bytes=size,
                    retained=True,
                    retention_class=referrers.get(
                        digest,
                        DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
                    ),
                    reason="reachable from a live lifecycle object",
                )
            )
        elif digest in superseded:
            entries.append(
                GCEntry(
                    digest=digest,
                    size_bytes=size,
                    retained=False,
                    retention_class=DiagnosticRetentionClass.CACHE,
                    reason="reachable only from superseded generations",
                )
            )
        else:
            entries.append(
                GCEntry(
                    digest=digest,
                    size_bytes=size,
                    retained=False,
                    retention_class=DiagnosticRetentionClass.CACHE,
                    reason="no registry reference",
                )
            )
    return _summarize(root, entries)


def _summarize(root: Path, entries: list[GCEntry]) -> GCPlan:
    total = sum(entry.size_bytes for entry in entries)
    retained = [entry for entry in entries if entry.retained]
    reclaimable = [entry for entry in entries if not entry.retained]
    return GCPlan(
        store_root=str(root),
        total_blobs=len(entries),
        total_bytes=total,
        retained_count=len(retained),
        retained_bytes=sum(entry.size_bytes for entry in retained),
        reclaimable_count=len(reclaimable),
        reclaimable_bytes=sum(entry.size_bytes for entry in reclaimable),
        entries=tuple(entries),
    )


def run_gc(
    store_root_path: Path | None = None,
    *,
    delete: bool = False,
) -> GCPlan:
    """Return a GC plan, deleting reclaimable blobs only when requested.

    When ``delete`` is true, reachability is recomputed immediately before any
    removal and a still-reachable blob refuses the entire operation.
    """
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    plan = plan_gc(root)
    if not delete:
        return plan
    live, _ = compute_reachable_blobs(root)
    refused = [
        entry.digest
        for entry in plan.entries
        if not entry.retained and entry.digest in live
    ]
    if refused:
        raise GCRefusedError(
            "diagnostic_gc_refused: "
            f"{len(refused)} blobs became reachable since planning",
        )
    for entry in plan.entries:
        if not entry.retained:
            blob_path(entry.digest, root).unlink(missing_ok=True)
    return plan


__all__ = [
    "GCEntry",
    "GCPlan",
    "GCRefusedError",
    "compute_reachable_blobs",
    "plan_gc",
    "run_gc",
]
