# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validate bounds and seal the complete staged analysis artifact set.

The public API delegates publication here; tests exercise both this boundary and
complete callable-to-publication runs. No missing access proof is synthesized.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from solar.analysis.access_contract import read_external_report
from solar.analysis.views import read_analysis_views
from solar.artifacts import sha256_file
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.contracts import (
    FORMAL_BOUND_KIND,
    SOL_BOUND_KINDS,
    ArtifactRef,
    SOLBound,
)
from solar.graph.contracts import OPERATOR_GRAPH_FILENAME
from solar.ir.access_artifact import ACCESS_FILENAME
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION


def extract_bound(
    analysis: Mapping[str, object],
    require_orojenesis: bool = True,
) -> SOLBound:
    """Accept a current, proven, finite analysis bound at the publication edge."""
    if analysis.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("formal analysis uses an unsupported schema")
    total = analysis.get("total") or {}
    metadata = analysis.get("metadata") or {}
    if not isinstance(total, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("analysis totals and metadata must be mappings")
    read_external_report(
        cast(Mapping[str, object], metadata), require_proof=True
    )
    read_analysis_views(metadata.get("views"))
    seconds = total.get("lower_bound_seconds")
    kind = str(metadata.get("bound_kind", ""))
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0
    ):
        raise ValueError("formal analysis lacks a finite lower bound")
    if require_orojenesis:
        if kind != FORMAL_BOUND_KIND:
            raise ValueError(
                f"strict analysis returned non-tile-aware bound kind {kind!r}",
            )
    elif kind not in SOL_BOUND_KINDS:
        raise ValueError(f"analysis returned unsupported bound kind {kind!r}")
    resource = total.get("compute_resource")
    return SOLBound(float(seconds), kind, str(resource) if resource else None)


def finish_artifacts(
    staging: Path,
    analysis: dict[str, object],
    graph_name: str,
) -> tuple[ArtifactRef, ...]:
    """Finalize analysis and hash the complete required publication inventory."""
    metadata = analysis.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("analysis metadata must be a mapping")
    cast(dict[str, object], metadata)["source_graph"] = graph_name
    analysis["metadata"] = metadata
    old_path = staging / "analysis.yaml"
    analysis_path = staging / ANALYSIS_FILENAME
    old_path.unlink(missing_ok=True)
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    required = {
        ACCESS_FILENAME,
        OPERATOR_GRAPH_FILENAME,
        graph_name,
        CONVERSION_ATTESTATION_FILENAME,
        ANALYSIS_FILENAME,
    }
    paths = _publication_files(staging)
    missing = sorted(
        required - {path.relative_to(staging).as_posix() for path in paths}
    )
    if missing:
        raise ValueError(f"analysis is missing required artifacts: {missing}")
    return tuple(
        ArtifactRef(path.relative_to(staging).as_posix(), sha256_file(path))
        for path in paths
    )


def _publication_files(staging: Path) -> tuple[Path, ...]:
    entries = tuple(staging.rglob("*"))
    symlinks = sorted(
        path.relative_to(staging).as_posix()
        for path in entries
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(
            f"analysis staging tree contains symbolic links: {symlinks}",
        )
    return tuple(
        sorted(
            (
                path
                for path in entries
                if path.is_file() and path.name != MANIFEST_FILENAME
            ),
            key=lambda path: path.relative_to(staging).as_posix(),
        ),
    )


def verify_publication_tree(
    staging: Path,
    artifacts: tuple[ArtifactRef, ...],
) -> None:
    """Reject a changed staged inventory before its atomic publication."""
    manifest_path = staging / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("analysis staging tree lacks a regular manifest")
    expected = {artifact.path for artifact in artifacts} | {MANIFEST_FILENAME}
    actual = {
        path.relative_to(staging).as_posix()
        for path in _publication_files(staging)
    } | {MANIFEST_FILENAME}
    if actual != expected:
        raise ValueError(
            "analysis staging tree changed after artifact hashing",
        )
