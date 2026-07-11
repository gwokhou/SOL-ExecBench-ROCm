# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Authoritative release-baseline validation for official score consumers.

The compact ``scoring_baseline.v1`` payload deliberately remains usable for
derived reports.  An official score must additionally prove that its entry was
published in a release bundle and passed an independent rerun.  Keeping this
check here prevents callers from accidentally treating a positive latency in a
compact artifact as release authority.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.scoring.baseline_artifact import (
    ScoringBaselineArtifact,
    load_scoring_baseline_artifact,
)

from .models import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    load_release_baseline_bundle,
    release_baseline_verification_from_dict,
    sha256_file,
)


@dataclass(frozen=True)
class OfficialReleaseBaseline:
    """A release baseline whose official rows passed independent verification."""

    baseline: ScoringBaselineArtifact
    bundle: ReleaseBaselineBundle
    verification: ReleaseBaselineVerification
    official_keys: frozenset[tuple[str, str]]

    def permits(
        self, definition: str, workload_uuid: str, latency_ms: float | None
    ) -> bool:
        """Return whether this exact compact-baseline value is official."""
        if (definition, workload_uuid) not in self.official_keys:
            return False
        entry = self.baseline.lookup(definition, workload_uuid)
        return (
            entry is not None
            and latency_ms is not None
            and math.isfinite(latency_ms)
            and entry.latency_ms == latency_ms
        )


def load_official_release_baseline(
    *,
    baseline_path: Path,
    bundle_path: Path,
    verification_path: Path,
) -> OfficialReleaseBaseline:
    """Load a release baseline and reject incomplete or mismatched evidence."""
    baseline_path = Path(baseline_path)
    bundle_path = Path(bundle_path)
    verification_path = Path(verification_path)
    baseline = load_scoring_baseline_artifact(
        baseline_path,
        required_schema_version="sol_execbench.scoring_baseline.v1",
    )
    bundle = load_release_baseline_bundle(bundle_path)
    verification = release_baseline_verification_from_dict(
        json.loads(verification_path.read_text(encoding="utf-8"))
    )

    if bundle.baseline_artifact_sha256 != sha256_file(baseline_path):
        raise ValueError("release baseline bundle does not match scoring baseline")
    if verification.release != bundle.release:
        raise ValueError("release baseline verification release does not match bundle")
    if verification.bundle_sha256 != sha256_file(bundle_path):
        raise ValueError("release baseline verification does not match bundle")

    bundle_rows = {row.key: row for row in bundle.workloads}
    verification_rows = {row.key: row for row in verification.workloads}
    if set(bundle_rows) != set(verification_rows):
        raise ValueError(
            "release baseline verification workload set does not match bundle"
        )

    official_keys: set[tuple[str, str]] = set()
    for key, row in bundle_rows.items():
        verified = verification_rows[key]
        if row.classification != "official":
            continue
        if (
            verified.original_classification != "official"
            or verified.classification != "official"
            or not verified.passed
        ):
            raise ValueError(
                "official release baseline workload lacks a passing official rerun: "
                f"{key[0]}:{key[1]}"
            )
        entry = baseline.lookup(*key)
        if entry is None or row.latency_ms is None:
            raise ValueError(
                "official release baseline workload is absent from scoring baseline: "
                f"{key[0]}:{key[1]}"
            )
        if entry.latency_ms != row.latency_ms:
            raise ValueError(
                "scoring baseline latency differs from release bundle for "
                f"{key[0]}:{key[1]}"
            )
        official_keys.add(key)

    return OfficialReleaseBaseline(
        baseline=baseline,
        bundle=bundle,
        verification=verification,
        official_keys=frozenset(official_keys),
    )
