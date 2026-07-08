# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

from sol_execbench.core.reports.claim_upgrade_evaluation import build_claim_upgrade_report
from sol_execbench.core.reports.claim_upgrade_models import (
    CLAIM_BOUNDARY_TEXT,
    CLAIM_LEVELS,
    CLAIM_UPGRADE_SCHEMA_VERSION,
    SOURCE_CHECKSUM_KEYS,
    ClaimEvaluation,
    ClaimRule,
    ClaimSourceRef,
    ClaimUpgradeClaimBoundary,
    ClaimUpgradeReport,
    ClaimUpgradeSources,
    PaperDenominatorClaimView,
)
from sol_execbench.core.reports.claim_upgrade_rendering import (
    render_claim_upgrade_markdown,
    write_claim_upgrade_reports,
)
from sol_execbench.core.reports.claim_upgrade_rules import default_claim_rules
from sol_execbench.core.reports.trust_summary import load_json as load_json, utc_timestamp

__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "CLAIM_LEVELS",
    "CLAIM_UPGRADE_SCHEMA_VERSION",
    "SOURCE_CHECKSUM_KEYS",
    "ClaimEvaluation",
    "ClaimRule",
    "ClaimSourceRef",
    "ClaimUpgradeClaimBoundary",
    "ClaimUpgradeReport",
    "ClaimUpgradeSources",
    "PaperDenominatorClaimView",
    "build_claim_upgrade_report",
    "default_claim_rules",
    "load_json",
    "render_claim_upgrade_markdown",
    "utc_timestamp",
    "write_claim_upgrade_reports",
]
