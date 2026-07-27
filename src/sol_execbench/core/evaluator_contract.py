# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable ownership boundary for evaluator consumers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.trace import EvaluationStatus
from sol_execbench.core.integrity.schema_versions import (
    EVALUATOR_CONTRACT_SCHEMA_VERSION,
)

SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION = EVALUATOR_CONTRACT_SCHEMA_VERSION
SOL_EXECBENCH_CONTRACT_VERSION = "3.0"
SOL_EXECBENCH_RELEASE = "v3.0.0"


class EvaluatorContract(BaseModelWithDocstrings):
    """Current evaluator, corpus, SOLAR, and scoring ownership contract."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    schema_version: Literal["sol_execbench.evaluator_contract.v4"] = (
        SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION
    )
    contract_version: Literal["3.0"] = SOL_EXECBENCH_CONTRACT_VERSION
    release: Literal["v3.0.0"] = SOL_EXECBENCH_RELEASE
    capabilities: dict[str, str] = Field(default_factory=dict)
    evaluation_statuses: list[str]
    corpus: dict[str, Any]
    scoring: dict[str, Any]
    boundaries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible contract payload."""
        return self.model_dump(mode="json")


def build_evaluator_contract() -> EvaluatorContract:
    """Build the single current public contract."""
    return EvaluatorContract(
        schema_version=SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
        contract_version=SOL_EXECBENCH_CONTRACT_VERSION,
        release=SOL_EXECBENCH_RELEASE,
        capabilities=_capabilities(),
        evaluation_statuses=list(EvaluationStatus),
        corpus=_corpus_contract(),
        scoring=_scoring_contract(),
        boundaries=_ownership_boundaries(),
    )


def _capabilities() -> dict[str, str]:
    """Return the owner of each evaluator capability."""
    return {
        "evaluation.reference_preparation": "trusted_reference_worker",
        "evaluation.candidate_execution": "untrusted_candidate_worker",
        "evaluation.relative_metrics": "sol_execbench_outer_runtime",
        "evaluation.static_review": "deterministic_ast_rules_not_paper_llm_judge",
        "evidence.canonical_execution": "trace_jsonl",
        "evidence.evaluation_sidecars": "diagnostic_non_authoritative",
        "evidence.runtime_environment": "platform_observation_non_authoritative",
        "solar.graph_extraction": "solar.graph",
        "solar.einsum_conversion": "solar.einsum",
        "solar.conversion_verification": "solar.verification",
        "solar.formal_bound": "solar.analysis",
        "solar.bound_policy": "rocm_formal_requires_pinned_orojenesis",
        "corpus.construction": "aka_derived_authored_problem_set",
        "corpus.selection": "sol_execbench",
        "corpus.materialization": "sol_execbench",
        "baseline.generation": "trusted_reference_eager_release_plan_v1",
        "official_score": "manifest_gated_content_addressed_bundle_verifier",
    }


def _corpus_contract() -> dict[str, Any]:
    """Return corpus identity and coverage facts in the public contract."""
    return {
        "manifest": "problems/AMD_AKA/manifest.yaml",
        "source": "AMD-AGI/AgentKernelArena",
        "execution_targets": ["gfx942", "gfx1150", "gfx1200"],
        "formal_target": "gfx1200",
        "scored_problems": 35,
        "scored_workloads": 122,
        "selection": "static_dtype_filter_then_trusted_live_probe",
        "local_output": "problems/local/AMD_AKA/<gfx-target>/",
    }


def _scoring_contract() -> dict[str, Any]:
    """Return the formula, prerequisites, and prohibited scoring behavior."""
    return {
        "formula": "1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))",
        "official_publication": "manifest_gated_content_addressed_release_bundle",
        "official_verifier_available": True,
        "official_policy_source": "pinned_corpus_manifest",
        "official_producer_gate": "reviewed_orojenesis_mapper_allowlist",
        "official_release_source": "repository_release_bundle",
        "baseline_strategy": "trusted_reference_eager_v1",
        "incorrect_candidate": 0,
        "aggregation": "workload_mean_within_problem_then_equal_problem_mean_v1",
        "requires": [
            "T_b > T_SOL",
            "T_k >= T_SOL",
            "exact_scored_corpus_coverage",
            "one_architecture_identity",
            "verified_solar_artifact_hashes",
            "publisher_candidate_execution",
            "canonical_release_baseline",
        ],
        "forbids": ["clipping", "bound_substitution", "sentinel_aggregation"],
    }


def _ownership_boundaries() -> list[dict[str, Any]]:
    """Return the explicit SOLAR and evaluator responsibility boundary."""
    return [
        {
            "owner": "solar",
            "scope": [
                "operator_graph",
                "einsum_graph",
                "conversion_attestation",
                "formal_sol_bound",
            ],
            "forbidden_inputs": [
                "candidate_solution",
                "candidate_runtime",
                "baseline_runtime",
                "score",
                "corpus_selection",
            ],
        },
        {
            "owner": "sol_execbench",
            "scope": [
                "problem_schema",
                "input_generation",
                "reference_preparation",
                "candidate_evaluation",
                "relative_metrics",
                "public_corpus",
                "baseline_identity",
                "official_score",
            ],
            "solar_import_path": "sol_execbench.core.solar_bridge",
        },
    ]


__all__ = [
    "SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION",
    "SOL_EXECBENCH_CONTRACT_VERSION",
    "SOL_EXECBENCH_RELEASE",
    "EvaluatorContract",
    "build_evaluator_contract",
]
