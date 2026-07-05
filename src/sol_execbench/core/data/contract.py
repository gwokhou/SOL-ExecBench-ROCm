# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-free evaluator contract metadata for downstream consumers."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from .base_model import BaseModelWithDocstrings
from .trace import EvaluationStatus


SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION = "sol_execbench.evaluator_contract.v2"
SOL_EXECBENCH_CONTRACT_VERSION = "1.0"
SOL_EXECBENCH_RELEASE = "v1.42"


class EvaluatorContract(BaseModelWithDocstrings):
    """SOL-owned evaluator and baseline export compatibility contract."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    schema_version: str
    """Schema identifier for the contract payload."""
    contract_version: str
    """Semantic contract version for HIP/SOL compatibility checks."""
    sol_release: str
    """SOL release that emitted this contract payload."""
    capabilities: dict[str, str] = Field(default_factory=dict)
    """Named capability tokens mapped to requirement levels."""
    trace_field_requirements: dict[str, list[str]]
    """Required trace field groups from the canonical trace JSONL contract."""
    correctness_fields: list[str]
    """Correctness metric field names owned by SOL evaluation."""
    timing_fields: list[str]
    """Timing metric field names owned by SOL evaluation."""
    scoring_fields: list[str]
    """Scoring and score-evidence field names available to consumers."""
    evaluation_statuses: list[str]
    """SOL evaluation status vocabulary."""
    failure_categories: dict[str, list[str]]
    """Consumer-facing failure buckets mapped from SOL status vocabulary."""
    baseline_export_fields: dict[str, object]
    """Measured baseline and scoring baseline field groups."""
    compatibility_metadata_fields: list[str]
    """Metadata fields consumers can persist for compatibility diagnostics."""
    boundaries: list[dict[str, object]]
    """Structured authority boundaries that keep benchmark truth in SOL."""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible contract payload."""
        return self.model_dump(mode="json")


def build_evaluator_contract() -> EvaluatorContract:
    """Build the current SOL evaluator compatibility contract."""

    return EvaluatorContract(
        schema_version=SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
        contract_version=SOL_EXECBENCH_CONTRACT_VERSION,
        sol_release=SOL_EXECBENCH_RELEASE,
        capabilities={
            "trace.correctness": "always",
            "trace.timing": "always",
            "trace.scoring": "always",
            "baseline.measured_export": "always",
            "baseline.scoring_artifact": "always",
            "compatibility.metadata": "always",
            "failure_categories": "always",
            "runtime.evidence": "optional",
            "profiling.evidence": "optional",
            "toolchain.routing": "optional",
            "static_kernel.evidence": "optional",
            "agent_feedback.sidecar": "profile:diagnostic",
            "profile_summary.sidecar": "profile:diagnostic",
        },
        trace_field_requirements={
            "top_level": ["definition", "workload", "solution", "evaluation"],
            "evaluation": [
                "status",
                "environment",
                "timestamp",
                "log",
                "correctness",
                "performance",
            ],
            "environment": ["hardware", "libs"],
        },
        correctness_fields=[
            "max_relative_error",
            "max_absolute_error",
            "has_nan",
            "has_inf",
            "extra",
        ],
        timing_fields=[
            "latency_ms",
            "reference_latency_ms",
            "speedup_factor",
        ],
        scoring_fields=[
            "status",
            "latency_ms",
            "reference_latency_ms",
            "speedup_factor",
            "baseline_latency_ms",
            "score",
            "score_source",
        ],
        evaluation_statuses=[status.value for status in EvaluationStatus],
        failure_categories={
            "passed": [EvaluationStatus.PASSED.value],
            "invalid_reference": [EvaluationStatus.INVALID_REFERENCE.value],
            "incorrect": [
                EvaluationStatus.INCORRECT_SHAPE.value,
                EvaluationStatus.INCORRECT_NUMERICAL.value,
                EvaluationStatus.INCORRECT_DTYPE.value,
            ],
            "execution_failure": [
                EvaluationStatus.RUNTIME_ERROR.value,
                EvaluationStatus.COMPILE_ERROR.value,
                EvaluationStatus.TIMEOUT.value,
            ],
            "policy_violation": [EvaluationStatus.REWARD_HACK.value],
        },
        baseline_export_fields={
            "measured_registry_schema_version": (
                "sol_execbench.measured_baseline_registry.v1"
            ),
            "measured_registry": [
                "schema_version",
                "generated_at",
                "sol_version",
                "source",
                "baseline_coverage_status",
                "coverage",
                "entries",
                "blockers",
            ],
            "measured_registry_entry": [
                "target_id",
                "definition",
                "workload_uuid",
                "latency_ms",
                "source",
                "hardware",
                "timestamp",
            ],
            "scoring_artifact_schema_version": "sol_execbench.scoring_baseline.v1",
            "scoring_artifact": [
                "schema_version",
                "derived",
                "release",
                "source",
                "summary",
                "entries",
            ],
            "scoring_artifact_entry": [
                "definition",
                "workload_uuid",
                "latency_ms",
                "solution",
                "source",
            ],
        },
        compatibility_metadata_fields=[
            "schema_version",
            "contract_version",
            "capabilities",
            "compatible",
            "blockers",
            "warnings",
            "source",
            "command",
            "payload_sha256",
        ],
        boundaries=[
            {"owner": "sol", "scope": "correctness", "immutable": True},
            {"owner": "sol", "scope": "timing", "immutable": True},
            {"owner": "sol", "scope": "scoring", "immutable": True},
            {"owner": "sol", "scope": "agent_feedback", "authority": "diagnostic"},
            {"owner": "sol", "scope": "profile_summary", "authority": "diagnostic"},
        ],
    )
