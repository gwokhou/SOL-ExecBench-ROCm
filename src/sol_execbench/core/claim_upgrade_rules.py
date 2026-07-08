# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

from sol_execbench.core.claim_upgrade_models import ClaimRule


def default_claim_rules() -> list[ClaimRule]:
    return [
        ClaimRule(
            claim_level="diagnostic_only",
            required_sources=[],
            required_conditions=["report_generated"],
        ),
        ClaimRule(
            claim_level="container_validated",
            required_sources=[
                "execution_closure",
                "paper_denominator",
                "consistency_report",
            ],
            required_conditions=[
                "closure_attempted_evidence",
                "denominator_present",
                "no_consistency_blockers",
            ],
        ),
        ClaimRule(
            claim_level="native_host_validated",
            required_sources=["hardware_validation", "matrix_report"],
            required_conditions=[
                "native_host_validation_evidence",
                "matrix_runtime_available",
            ],
        ),
        ClaimRule(
            claim_level="score_authoritative",
            required_sources=[
                "amd_score_report",
                "amd_sol_report",
                "solar_derivation",
                "evaluation_stability",
                "consistency_report",
            ],
            required_conditions=[
                "score_evidence_present",
                "amd_sol_evidence_present",
                "solar_derivation_present",
                "stable_timing",
                "no_consistency_blockers",
            ],
        ),
        ClaimRule(
            claim_level="paper_parity_candidate",
            required_sources=[
                "paper_denominator",
                "amd_score_report",
                "amd_sol_report",
                "solar_derivation",
                "amd_bound_sanity",
                "hardware_validation",
            ],
            required_conditions=[
                "full_suite_accounted",
                "score_evidence_present",
                "amd_sol_evidence_present",
                "solar_derivation_present",
                "bound_sanity_clean",
                "native_host_validation_evidence",
            ],
        ),
        ClaimRule(
            claim_level="leaderboard_ready",
            required_sources=["paper_denominator", "hardware_validation"],
            required_conditions=[
                "paper_parity_candidate",
                "submission_policy_exists",
                "hosted_service_evidence",
            ],
        ),
    ]
