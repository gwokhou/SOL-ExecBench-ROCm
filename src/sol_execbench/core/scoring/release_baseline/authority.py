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
from sol_execbench.core.scoring.amd_hardware_models import load_amd_hardware_model
from sol_execbench.core.scoring.amd_sol import amd_sol_bound_from_dict
from sol_execbench.core.integrity.checksums import sha256_file

from .models import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    load_release_baseline_bundle,
    release_baseline_verification_from_dict,
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

    def verifies_bound_reference(
        self,
        definition: str,
        workload_uuid: str,
        sol_bound_ref: str | None,
        hardware_model_ref: str | None,
    ) -> bool:
        """Verify immutable current bound and model inputs for one official score.

        The score report is derived data.  Official authority therefore requires
        that its cited inputs resolve to the exact, checksummed files published
        in the paired release bundle.  This deliberately makes retired v2 bound
        evidence ineligible after the v3 fusion-model migration.
        """
        row = next(
            (
                row
                for row in self.bundle.workloads
                if row.key == (definition, workload_uuid)
            ),
            None,
        )
        if (
            row is None
            or row.bound_ref is None
            or row.bound_sha256 is None
            or row.hardware_model_ref is None
            or row.hardware_model_sha256 is None
            or sol_bound_ref != row.bound_ref
            or hardware_model_ref != row.hardware_model_ref
        ):
            return False
        bound_path = Path(row.bound_ref)
        hardware_path = Path(row.hardware_model_ref)
        if not bound_path.is_file() or not hardware_path.is_file():
            return False
        if (
            sha256_file(bound_path) != row.bound_sha256
            or sha256_file(hardware_path) != row.hardware_model_sha256
        ):
            return False
        try:
            bound_payload = json.loads(bound_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(bound_payload, dict):
            return False
        try:
            bound = amd_sol_bound_from_dict(bound_payload)
            model = load_amd_hardware_model(hardware_path)
        except (KeyError, OSError, TypeError, ValueError):
            return False
        if (
            bound_payload.get("definition") != definition
            or bound_payload.get("workload_uuid") != workload_uuid
        ):
            return False
        if bound.hardware_model.architecture != model.architecture:
            return False
        fusion_path = Path(bound.fusion_validation_ref)
        if not fusion_path.is_absolute():
            fusion_path = bound_path.parent / fusion_path
        if (
            not fusion_path.is_file()
            or sha256_file(fusion_path) != bound.fusion_validation_sha256
        ):
            return False
        return True


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
    rerun_trace_path = Path(verification.rerun_trace_ref)
    if (
        not rerun_trace_path.is_file()
        or sha256_file(rerun_trace_path) != verification.rerun_trace_sha256
    ):
        raise ValueError("release baseline rerun trace does not match verification")
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
        if (
            row.trace_ref is None
            or row.trace_sha256 is None
            or not Path(row.trace_ref).is_file()
            or sha256_file(Path(row.trace_ref)) != row.trace_sha256
        ):
            raise ValueError(
                "official release baseline trace does not match bundle for "
                f"{key[0]}:{key[1]}"
            )
        official_keys.add(key)

    return OfficialReleaseBaseline(
        baseline=baseline,
        bundle=bundle,
        verification=verification,
        official_keys=frozenset(official_keys),
    )
