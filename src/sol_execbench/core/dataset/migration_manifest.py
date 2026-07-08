# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration manifest construction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .manifest import utc_timestamp
from .migration_models import (
    SOL_SOURCE_ID,
    DatasetMigrationManifest,
    MigrationArtifact,
    MigrationBlocker,
    MigrationDenominators,
    MigrationLicenseBoundary,
    MigrationSource,
)


def build_manifest(
    *,
    migration_kind: Literal["sol_execbench", "flashinfer_trace"],
    source_id: str,
    repo_id: str,
    source_root: Path,
    output_root: Path,
    selected_categories: tuple[str, ...],
    artifacts: list[MigrationArtifact],
    blockers: list[MigrationBlocker],
    discovered: int,
    migrated: int,
    source_revision: str | None,
    created_at: str | None,
) -> DatasetMigrationManifest:
    if source_id == SOL_SOURCE_ID:
        boundary = MigrationLicenseBoundary(
            source_boundary=source_id,
            license="NVIDIA Evaluation Dataset License",
            redistribution_class="excluded",
            repository_redistribution=False,
            release_bundle_redistribution=False,
            attribution=(
                "NVIDIA SOL-ExecBench; obtain from upstream under NVIDIA "
                "Evaluation Dataset License terms."
            ),
        )
    else:
        boundary = MigrationLicenseBoundary(
            source_boundary=source_id,
            license="Apache-2.0",
            redistribution_class="publishable",
            repository_redistribution=True,
            release_bundle_redistribution=True,
            attribution=(
                "FlashInfer Trace by flashinfer-ai/flashinfer-trace under "
                "Apache-2.0; retain required notices when redistributing."
            ),
        )

    ordered_artifacts = sorted(
        artifacts, key=lambda item: (item.output_ref, item.kind, item.source_ref)
    )
    ordered_blockers = sorted(
        blockers,
        key=lambda item: (
            item.problem_id or "",
            item.code,
            item.source_ref or "",
            item.output_ref or "",
            item.message,
        ),
    )
    denominators = MigrationDenominators(
        discovered_problems=discovered,
        migrated_problems=migrated,
        generated_artifacts=len(ordered_artifacts),
        blockers=sum(
            1 for blocker in ordered_blockers if blocker.severity == "blocker"
        ),
        warnings=sum(
            1 for blocker in ordered_blockers if blocker.severity == "warning"
        ),
    )
    return DatasetMigrationManifest(
        created_at=created_at or utc_timestamp(),
        migration_kind=migration_kind,
        source=MigrationSource(
            source_id=source_id,
            repo_id=repo_id,
            revision=source_revision,
            source_root=Path(source_root).as_posix(),
        ),
        output_root=Path(output_root).as_posix(),
        selected_categories=selected_categories,
        license_boundary=boundary,
        artifacts=ordered_artifacts,
        blockers=ordered_blockers,
        denominators=denominators,
    ).with_checksum()
