# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""AMD SOL/SOLAR bound sanity diagnostic sidecar helpers."""

from __future__ import annotations

from .amd_bound_sanity_builder import build_amd_bound_sanity_report
from .amd_bound_sanity_helpers import (  # noqa: F401
    _add_gap,
    _add_gap_group,
    _apply_missing_required_artifact_gaps,
    _artifact_uuid,
    _checksum,
    _contains_provisional,
    _contains_unsupported,
    _coverage_summary,
    _dict_value,
    _ensure_workload,
    _extend_unique,
    _is_degraded_status,
    _next_evidence,
    _optional_str,
    _payload_artifacts,
    _provisional_artifact,
    _sorted_evidence_gaps,
    _sorted_jsonable,
    _source,
    _source_from_ref,
    _suite_warnings,
    _warnings_from,
    _workload_ref,
    _workload_seed,
)
from .amd_bound_sanity_io import load_json, write_amd_bound_sanity_reports
from .amd_bound_sanity_models import (
    AMD_BOUND_SANITY_SCHEMA_VERSION,
    CLAIM_BOUNDARY_TEXT,
    PRIMARY_STATUS_ORDER,
    SANITY_STATUS_KEYS,
    SOURCE_CHECKSUM_KEYS,
    AmdBoundSanityArtifactAvailability,
    AmdBoundSanityClaimBoundary,
    AmdBoundSanityEvidenceGap,
    AmdBoundSanityReport,
    AmdBoundSanitySourceRef,
    AmdBoundSanitySourceStatuses,
    AmdBoundSanitySources,
    AmdBoundSanityStatusTotals,
    AmdBoundSanityWorkload,
)
from .amd_bound_sanity_rendering import (  # noqa: F401
    _source_rows,
    render_amd_bound_sanity_markdown,
)

__all__ = [
    "AMD_BOUND_SANITY_SCHEMA_VERSION",
    "AmdBoundSanityArtifactAvailability",
    "AmdBoundSanityClaimBoundary",
    "AmdBoundSanityEvidenceGap",
    "AmdBoundSanityReport",
    "AmdBoundSanitySourceRef",
    "AmdBoundSanitySourceStatuses",
    "AmdBoundSanitySources",
    "AmdBoundSanityStatusTotals",
    "AmdBoundSanityWorkload",
    "CLAIM_BOUNDARY_TEXT",
    "PRIMARY_STATUS_ORDER",
    "SANITY_STATUS_KEYS",
    "SOURCE_CHECKSUM_KEYS",
    "build_amd_bound_sanity_report",
    "load_json",
    "render_amd_bound_sanity_markdown",
    "write_amd_bound_sanity_reports",
]
