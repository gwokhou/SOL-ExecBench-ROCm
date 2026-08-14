# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current benchmark input and evaluator artifact schemas."""

from enum import StrEnum


class BenchmarkArtifactSchema(StrEnum):
    """Canonical benchmark contract identifiers."""

    BENCHMARK_CONFIG = "sol_execbench.benchmark_config.v2"
    DEFINITION = "sol_execbench.definition.v1"
    EVALUATOR_CONTRACT = "sol_execbench.evaluator_contract.v6"
    SOLUTION = "sol_execbench.solution.v1"
    TRACE = "sol_execbench.trace.v1"
    WORKLOAD = "sol_execbench.workload.v2"


__all__ = ["BenchmarkArtifactSchema"]
