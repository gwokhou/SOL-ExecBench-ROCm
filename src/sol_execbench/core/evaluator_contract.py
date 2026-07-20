# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable ownership boundary for evaluator consumers."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.trace import EvaluationStatus

SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION = "sol_execbench.evaluator_contract.v3"
SOL_EXECBENCH_CONTRACT_VERSION = "3.0"
SOL_EXECBENCH_RELEASE = "v3.0.0"


class EvaluatorContract(BaseModelWithDocstrings):
    """Current evaluator, corpus, SOLAR, and scoring ownership contract."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    schema_version: str
    contract_version: str
    release: str
    capabilities: dict[str, str] = Field(default_factory=dict)
    evaluation_statuses: list[str]
    corpus: dict[str, Any]
    scoring: dict[str, Any]
    boundaries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible contract payload."""
        return self.model_dump(mode="json")


def build_evaluator_contract() -> EvaluatorContract:
    """Build the single current public contract; v1/v2 are unsupported."""
    return EvaluatorContract(
        schema_version=SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
        contract_version=SOL_EXECBENCH_CONTRACT_VERSION,
        release=SOL_EXECBENCH_RELEASE,
        capabilities={
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
            "corpus.construction": "not_implemented_uses_pinned_upstream_dataset",
            "corpus.selection": "sol_execbench",
            "corpus.materialization": "sol_execbench",
            "baseline.generation": "not_implemented",
            "official_score": "unavailable_status_only_no_scorer",
        },
        evaluation_statuses=[status.value for status in EvaluationStatus],
        corpus={
            "manifest": "problems/RX_9060_XT/manifest.yaml",
            "dataset_id": "nvidia/SOL-ExecBench",
            "dataset_revision": "63699402f003496acc3af4eb534a5304a8ac1ea9",
            "formal_target": "gfx1200",
            "scored_workloads": 14,
            "compatibility_sentinels": 1,
            "local_output": "problems/local/",
        },
        scoring={
            "formula": "1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))",
            "official_authority": "unavailable_release_authority_not_published",
            "scorer_implemented": False,
            "incorrect_candidate": 0,
            "aggregation": ("workload_mean_within_problem_then_equal_problem_mean_v1"),
            "requires": [
                "T_b > T_SOL",
                "T_k >= T_SOL",
                "exact_scored_corpus_coverage",
                "one_architecture_identity",
                "verified_solar_artifact_hashes",
                "trusted_candidate_execution_attestation",
                "independent_release_baseline_rerun",
            ],
            "forbids": ["clipping", "bound_substitution", "sentinel_aggregation"],
        },
        boundaries=[
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
        ],
    )


__all__ = [
    "SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION",
    "SOL_EXECBENCH_CONTRACT_VERSION",
    "SOL_EXECBENCH_RELEASE",
    "EvaluatorContract",
    "build_evaluator_contract",
]
